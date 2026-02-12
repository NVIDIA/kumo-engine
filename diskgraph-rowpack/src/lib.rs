use anyhow::{anyhow, Result};
use arrow_array::{Array, RecordBatch};
use base64::Engine;
use chrono::{DateTime, Utc};
use gxhash::{HashMap, HashMapExt};
use half::f16;
use half::slice::HalfFloatSliceExt;
use num_bigint::BigInt;
use num_traits::Signed;
use num_traits::ToPrimitive;
use rmp::encode as mp;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use uuid_simd::{format_simple, AsOut, AsciiCase};

pub mod access;
pub mod parser;
#[cfg(test)]
mod tests_arrow;
#[cfg(test)]
#[allow(clippy::cloned_ref_to_slice_refs)]
mod tests_arrow_matrix;
#[cfg(test)]
mod tests_casting;
#[cfg(test)]
mod tests_type_matrix;
#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests_unit;

pub use crate::access::{BinaryStringHint, RowAccess, RowAccessHints};
use crate::access::{FloatListValues, FloatListView};

pub(crate) const ROWPACK_VERSION: u8 = 3;
pub(crate) const MAX_INPUT_STRING_BYTES: usize = 64 * 1024 * 1024;
const MAX_FIELD_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const MAX_ARRAY_LEN: usize = 1_000_000;
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
                    next.extend(std::iter::repeat_n(-1, 256));
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

#[derive(Default)]
pub struct RowAccessStats<'a> {
    pub trimmed_strings: Option<&'a std::sync::atomic::AtomicU64>,
}

impl<'a> RowAccessStats<'a> {
    pub fn new(trimmed_strings: Option<&'a std::sync::atomic::AtomicU64>) -> Self {
        Self { trimmed_strings }
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

pub(crate) fn write_varint(mut value: u64, out: &mut Vec<u8>) {
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
        out.extend_from_slice(&nanos.to_be_bytes());
        out.extend_from_slice(&secs.to_be_bytes());
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
    out.extend_from_slice(&nanos.to_be_bytes());
    out.extend_from_slice(&secs.to_be_bytes());
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
                    if access
                        .with_string(idx, |v| {
                            stats.observe_string(v, &self.config);
                            Ok(())
                        })?
                        .is_none()
                    {
                        stats.observe_null()
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
                    if access
                        .with_float32_array(idx, |values| {
                            stats.observe_array_f32(values);
                            Ok(())
                        })?
                        .is_none()
                    {
                        stats.observe_null()
                    }
                }
                IngestedType::Float64Array => {
                    if access
                        .with_float64_array(idx, |values| {
                            stats.observe_array_f64(values);
                            Ok(())
                        })?
                        .is_none()
                    {
                        stats.observe_null()
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
        let tag_bytes = (self.columns.len() * 2).div_ceil(8);
        let bool_bytes = bools.div_ceil(8);
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
                (*bits).div_ceil(8) as usize
            }
        }
        ColumnEncoding::Date { varint, bits } => {
            if *varint {
                varint_bytes(*bits)
            } else {
                (*bits).div_ceil(8) as usize
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
                let packed = (avg * bits as usize).div_ceil(8);
                varint_bytes_u64(avg as u64) + packed
            }
            StringEncoding::Bpe { .. } => {
                let avg = avg_len(stats.string_bytes, stats.string_total);
                let bits = bpe_token_bits(match encoding {
                    StringEncoding::Bpe { dict } => dict.len(),
                    _ => 0,
                }) as usize;
                let packed = (avg * bits).div_ceil(8);
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
    (bits as usize).div_ceil(7)
}

fn varint_bytes_u64(value: u64) -> usize {
    let bits = 64 - value.leading_zeros() as usize;
    (bits.max(1)).div_ceil(7)
}

fn varint_bytes_u128(value: u128) -> usize {
    let bits = 128 - value.leading_zeros() as usize;
    (bits.max(1)).div_ceil(7)
}

fn int_bits(min: i64, max: i64) -> u8 {
    let max_abs = min.unsigned_abs().max(max.unsigned_abs());
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

pub(crate) fn decimal_raw_bytes(precision: u32) -> Result<u8> {
    if precision == 0 {
        return Err(anyhow!("rowpack decimal precision missing"));
    }
    let bits = (precision as f64 * std::f64::consts::LOG2_10).ceil() as u32;
    let bytes = bits.div_ceil(8).max(1);
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
                if self.string_samples.len() < config.bpe_max_samples()
                    && self.string_sample_bytes + bytes.len() <= config.bpe_max_dict_bytes() * 4
                {
                    self.string_samples.push(bytes.to_vec());
                    self.string_sample_bytes += bytes.len();
                }
                if self.alphabet_ok {
                    if bytes.len() > config.string_long_min_len() {
                        self.alphabet_ok = false;
                    } else {
                        let mut ok = true;
                        for b in bytes {
                            let idx = *b as usize;
                            if !self.alphabet_bytes[idx]
                                && self.alphabet_len + 1 > ALPHABET_MAX_CHARS
                            {
                                ok = false;
                                break;
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
        if self.string_samples.len() < config.bpe_max_samples()
            && self.string_sample_bytes + bytes.len() <= config.bpe_max_dict_bytes() * 4
        {
            self.string_samples.push(bytes.to_vec());
            self.string_sample_bytes += bytes.len();
        }
        if self.alphabet_ok {
            if bytes.len() > config.string_long_min_len() {
                self.alphabet_ok = false;
            } else {
                let mut ok = true;
                for b in bytes {
                    let idx = *b as usize;
                    if !self.alphabet_bytes[idx] && self.alphabet_len + 1 > ALPHABET_MAX_CHARS {
                        ok = false;
                        break;
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
    let max_abs = min.unsigned_abs().max(max.unsigned_abs());
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

pub(crate) fn train_bpe_dict(
    samples: &[Vec<u8>],
    max_entries: usize,
    max_bytes: usize,
) -> Vec<Vec<u8>> {
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
    let mut entries: Vec<(Vec<u8>, usize)> = dict_entries.into_iter().zip(dict_counts).collect();
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
    f16_buf: Vec<f16>,
}

impl RowPackScratch {
    pub fn new(ctx: &RowPackContext) -> Self {
        let tag_bytes = (ctx.columns.len() * 2).div_ceil(8);
        let bool_columns = ctx
            .columns
            .iter()
            .filter(|c| matches!(c.encoding, ColumnEncoding::Bool))
            .count();
        let bool_bytes = bool_columns.div_ceil(8);
        Self {
            tags: BitWriter::with_capacity(tag_bytes),
            bools: BitWriter::with_capacity(bool_bytes),
            data: Vec::with_capacity(ctx.estimated_row_bytes as usize),
            bpe_tokens: Vec::new(),
            lz4_buf: Vec::new(),
            f16_buf: Vec::new(),
        }
    }

    fn reset(&mut self, ctx: &RowPackContext) {
        let tag_bytes = (ctx.columns.len() * 2).div_ceil(8);
        if self.tags.buf.capacity() < tag_bytes {
            self.tags.buf.reserve(tag_bytes - self.tags.buf.capacity());
        }
        let bool_columns = ctx
            .columns
            .iter()
            .filter(|c| matches!(c.encoding, ColumnEncoding::Bool))
            .count();
        let bool_bytes = bool_columns.div_ceil(8);
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
        self.f16_buf.clear();
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
                IngestedType::String =>
                {
                    #[allow(clippy::manual_unwrap_or_default)]
                    access
                        .with_string(idx, |v| {
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
                        })?
                        .unwrap_or(0u8)
                }
                IngestedType::Bytes =>
                {
                    #[allow(clippy::manual_unwrap_or_default)]
                    access
                        .with_bytes(idx, |v| {
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
                        })?
                        .unwrap_or(0u8)
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
                IngestedType::Float32Array => match access.float_list_view(idx)? {
                    None => 0u8,
                    Some(view) => {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            let f16_buf = &mut scratch.f16_buf;
                            encode_float_list_with_tag(
                                &mut scratch.data,
                                col.logical_type,
                                *storage,
                                view,
                                f16_buf,
                            )?
                        } else {
                            return Err(anyhow!("rowpack float32 array encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float16Array => match access.float_list_view(idx)? {
                    None => 0u8,
                    Some(view) => {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            let f16_buf = &mut scratch.f16_buf;
                            encode_float_list_with_tag(
                                &mut scratch.data,
                                col.logical_type,
                                *storage,
                                view,
                                f16_buf,
                            )?
                        } else {
                            return Err(anyhow!("rowpack float16 array encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float64Array => match access.float_list_view(idx)? {
                    None => 0u8,
                    Some(view) => {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            let f16_buf = &mut scratch.f16_buf;
                            encode_float_list_with_tag(
                                &mut scratch.data,
                                col.logical_type,
                                *storage,
                                view,
                                f16_buf,
                            )?
                        } else {
                            return Err(anyhow!("rowpack float64 array encoding mismatch"));
                        }
                    }
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

    pub fn encode_row_from_access_into_buffers(
        &self,
        access: &mut RowAccess<'_>,
        tags: &mut Vec<u8>,
        bools: &mut Vec<u8>,
        data: &mut Vec<u8>,
        f16_buf: &mut Vec<f16>,
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
                IngestedType::String =>
                {
                    #[allow(clippy::manual_unwrap_or_default)]
                    access
                        .with_string(idx, |v| {
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
                        })?
                        .unwrap_or(0u8)
                }
                IngestedType::Bytes =>
                {
                    #[allow(clippy::manual_unwrap_or_default)]
                    access
                        .with_bytes(idx, |v| {
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
                        })?
                        .unwrap_or(0u8)
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
                IngestedType::Float32Array => match access.float_list_view(idx)? {
                    None => 0u8,
                    Some(view) => {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            encode_float_list_with_tag(
                                data,
                                col.logical_type,
                                *storage,
                                view,
                                f16_buf,
                            )?
                        } else {
                            return Err(anyhow!("rowpack float32 array encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float16Array => match access.float_list_view(idx)? {
                    None => 0u8,
                    Some(view) => {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            encode_float_list_with_tag(
                                data,
                                col.logical_type,
                                *storage,
                                view,
                                f16_buf,
                            )?
                        } else {
                            return Err(anyhow!("rowpack float16 array encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float64Array => match access.float_list_view(idx)? {
                    None => 0u8,
                    Some(view) => {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            encode_float_list_with_tag(
                                data,
                                col.logical_type,
                                *storage,
                                view,
                                f16_buf,
                            )?
                        } else {
                            return Err(anyhow!("rowpack float64 array encoding mismatch"));
                        }
                    }
                },
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

#[allow(clippy::too_many_arguments)]
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
    let packed_len = (bytes.len() * map.bits as usize).div_ceil(8);
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
    bpe_tokens: Option<&mut Vec<u32>>,
) -> Result<u8> {
    let bytes = value.as_bytes();
    let trie = bpe.ok_or_else(|| anyhow!("rowpack missing bpe trie"))?;
    let mut local_tokens = Vec::new();
    let tokens = match bpe_tokens {
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

fn encode_string_lz4(out: &mut Vec<u8>, value: &str, lz4_buf: Option<&mut Vec<u8>>) -> Result<u8> {
    let bytes = value.as_bytes();
    let raw_len = bytes.len();
    let mut local_buf = Vec::new();
    let buf = match lz4_buf {
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

#[allow(clippy::too_many_arguments)]
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
            let tag = encode_float_slice_f32_with_tag(out, col.logical_type, *storage, values)?;
            Ok(tag)
        }
        (ColumnEncoding::FloatArray { storage }, ValueRef::Float64Array(values)) => {
            let tag = encode_float_slice_f64_with_tag(out, col.logical_type, *storage, values)?;
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

#[inline]
fn extend_f16_slice_le(out: &mut Vec<u8>, slice: &[f16]) {
    if cfg!(target_endian = "little") {
        let len = std::mem::size_of_val(slice);
        let ptr = slice.as_ptr() as *const u8;
        // SAFETY: f16 is Copy; we only read the underlying bytes.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        out.extend_from_slice(bytes);
    } else {
        for v in slice {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
}

#[inline]
fn extend_f32_slice_le(out: &mut Vec<u8>, slice: &[f32]) {
    if cfg!(target_endian = "little") {
        let len = std::mem::size_of_val(slice);
        let ptr = slice.as_ptr() as *const u8;
        // SAFETY: f32 is Copy; we only read the underlying bytes.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        out.extend_from_slice(bytes);
    } else {
        for v in slice {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

#[inline]
fn extend_f64_slice_le(out: &mut Vec<u8>, slice: &[f64]) {
    if cfg!(target_endian = "little") {
        let len = std::mem::size_of_val(slice);
        let ptr = slice.as_ptr() as *const u8;
        // SAFETY: f64 is Copy; we only read the underlying bytes.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        out.extend_from_slice(bytes);
    } else {
        for v in slice {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

fn encode_float_list_with_tag(
    out: &mut Vec<u8>,
    logical: IngestedType,
    storage: FloatStorage,
    view: FloatListView,
    f16_buf: &mut Vec<f16>,
) -> Result<u8> {
    let _ = logical;
    let len = view.end.saturating_sub(view.start);
    let bytes_per = match storage {
        FloatStorage::F16 => 2,
        FloatStorage::F32 => 4,
        FloatStorage::F64 => 8,
    };
    out.reserve(varint_bytes_u64(len as u64) + len * bytes_per);
    write_varint(len as u64, out);
    match view.values {
        FloatListValues::F16(values) => {
            let values_arr = values
                .as_any()
                .downcast_ref::<arrow_array::Float16Array>()
                .ok_or_else(|| anyhow!("float16 list cast"))?;
            let slice = &values_arr.values()[view.start..view.end];
            match storage {
                FloatStorage::F16 => {
                    if values_arr.null_count() == 0 {
                        extend_f16_slice_le(out, slice);
                    } else {
                        for idx in view.start..view.end {
                            if values_arr.is_null(idx) {
                                out.extend_from_slice(&0u16.to_le_bytes());
                            } else {
                                out.extend_from_slice(
                                    &values_arr.value(idx).to_bits().to_le_bytes(),
                                );
                            }
                        }
                    }
                }
                FloatStorage::F32 => {
                    if values_arr.null_count() == 0 {
                        for v in slice {
                            out.extend_from_slice(&f16::to_f32(*v).to_le_bytes());
                        }
                    } else {
                        for idx in view.start..view.end {
                            if values_arr.is_null(idx) {
                                out.extend_from_slice(&0f32.to_le_bytes());
                            } else {
                                out.extend_from_slice(
                                    &f16::to_f32(values_arr.value(idx)).to_le_bytes(),
                                );
                            }
                        }
                    }
                }
                FloatStorage::F64 => {
                    if values_arr.null_count() == 0 {
                        for v in slice {
                            out.extend_from_slice(&(f16::to_f32(*v) as f64).to_le_bytes());
                        }
                    } else {
                        for idx in view.start..view.end {
                            if values_arr.is_null(idx) {
                                out.extend_from_slice(&0f64.to_le_bytes());
                            } else {
                                out.extend_from_slice(
                                    &(f16::to_f32(values_arr.value(idx)) as f64).to_le_bytes(),
                                );
                            }
                        }
                    }
                }
            }
        }
        FloatListValues::F32(values) => {
            let values_arr = values
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .ok_or_else(|| anyhow!("float32 list cast"))?;
            let slice = &values_arr.values()[view.start..view.end];
            match storage {
                FloatStorage::F16 => {
                    if values_arr.null_count() == 0 {
                        f16_buf.resize(slice.len(), f16::from_bits(0));
                        f16_buf.as_mut_slice().convert_from_f32_slice(slice);
                        extend_f16_slice_le(out, f16_buf);
                    } else {
                        for idx in view.start..view.end {
                            if values_arr.is_null(idx) {
                                out.extend_from_slice(&0u16.to_le_bytes());
                            } else {
                                out.extend_from_slice(
                                    &f16::from_f32(values_arr.value(idx)).to_bits().to_le_bytes(),
                                );
                            }
                        }
                    }
                }
                FloatStorage::F32 => {
                    if values_arr.null_count() == 0 {
                        extend_f32_slice_le(out, slice);
                    } else {
                        for idx in view.start..view.end {
                            if values_arr.is_null(idx) {
                                out.extend_from_slice(&0f32.to_le_bytes());
                            } else {
                                out.extend_from_slice(&values_arr.value(idx).to_le_bytes());
                            }
                        }
                    }
                }
                FloatStorage::F64 => {
                    if values_arr.null_count() == 0 {
                        for v in slice {
                            out.extend_from_slice(&(*v as f64).to_le_bytes());
                        }
                    } else {
                        for idx in view.start..view.end {
                            if values_arr.is_null(idx) {
                                out.extend_from_slice(&0f64.to_le_bytes());
                            } else {
                                out.extend_from_slice(
                                    &(values_arr.value(idx) as f64).to_le_bytes(),
                                );
                            }
                        }
                    }
                }
            }
        }
        FloatListValues::F64(values) => {
            let values_arr = values
                .as_any()
                .downcast_ref::<arrow_array::Float64Array>()
                .ok_or_else(|| anyhow!("float64 list cast"))?;
            let slice = &values_arr.values()[view.start..view.end];
            match storage {
                FloatStorage::F16 => {
                    if values_arr.null_count() == 0 {
                        f16_buf.resize(slice.len(), f16::from_bits(0));
                        f16_buf.as_mut_slice().convert_from_f64_slice(slice);
                        extend_f16_slice_le(out, f16_buf);
                    } else {
                        for idx in view.start..view.end {
                            if values_arr.is_null(idx) {
                                out.extend_from_slice(&0u16.to_le_bytes());
                            } else {
                                out.extend_from_slice(
                                    &f16::from_f32(values_arr.value(idx) as f32)
                                        .to_bits()
                                        .to_le_bytes(),
                                );
                            }
                        }
                    }
                }
                FloatStorage::F32 => {
                    if values_arr.null_count() == 0 {
                        for v in slice {
                            out.extend_from_slice(&(*v as f32).to_le_bytes());
                        }
                    } else {
                        for idx in view.start..view.end {
                            if values_arr.is_null(idx) {
                                out.extend_from_slice(&0f32.to_le_bytes());
                            } else {
                                out.extend_from_slice(
                                    &(values_arr.value(idx) as f32).to_le_bytes(),
                                );
                            }
                        }
                    }
                }
                FloatStorage::F64 => {
                    if values_arr.null_count() == 0 {
                        extend_f64_slice_le(out, slice);
                    } else {
                        for idx in view.start..view.end {
                            if values_arr.is_null(idx) {
                                out.extend_from_slice(&0f64.to_le_bytes());
                            } else {
                                out.extend_from_slice(&values_arr.value(idx).to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(1)
}

fn encode_float_slice_f32_with_tag(
    out: &mut Vec<u8>,
    logical: IngestedType,
    storage: FloatStorage,
    values: &[f32],
) -> Result<u8> {
    let _ = logical;
    let bytes_per = match storage {
        FloatStorage::F16 => 2,
        FloatStorage::F32 => 4,
        FloatStorage::F64 => 8,
    };
    out.reserve(varint_bytes_u64(values.len() as u64) + values.len() * bytes_per);
    write_varint(values.len() as u64, out);
    match storage {
        FloatStorage::F16 => {
            for v in values {
                out.extend_from_slice(&f16::from_f32(*v).to_bits().to_le_bytes());
            }
        }
        FloatStorage::F32 => {
            extend_f32_slice_le(out, values);
        }
        FloatStorage::F64 => {
            for v in values {
                out.extend_from_slice(&(*v as f64).to_le_bytes());
            }
        }
    }
    Ok(1)
}

fn encode_float_slice_f64_with_tag(
    out: &mut Vec<u8>,
    logical: IngestedType,
    storage: FloatStorage,
    values: &[f64],
) -> Result<u8> {
    let _ = logical;
    let bytes_per = match storage {
        FloatStorage::F16 => 2,
        FloatStorage::F32 => 4,
        FloatStorage::F64 => 8,
    };
    out.reserve(varint_bytes_u64(values.len() as u64) + values.len() * bytes_per);
    write_varint(values.len() as u64, out);
    match storage {
        FloatStorage::F16 => {
            for v in values {
                out.extend_from_slice(&f16::from_f32(*v as f32).to_bits().to_le_bytes());
            }
        }
        FloatStorage::F32 => {
            for v in values {
                out.extend_from_slice(&(*v as f32).to_le_bytes());
            }
        }
        FloatStorage::F64 => {
            extend_f64_slice_le(out, values);
        }
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

pub(crate) fn bpe_token_bits(dict_len: usize) -> u8 {
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

pub(crate) fn write_tokens_packed(out: &mut Vec<u8>, tokens: &[u32], bits: u8) {
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
        let packed = (sample.len() * bits).div_ceil(8);
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
        let packed = (tokens.len() * bits).div_ceil(8);
        total += packed as u64;
    }
    total
}

#[derive(Clone)]
pub struct RowPackDecoder {
    ctx: RowPackContext,
    dict_values: Vec<Option<Vec<Vec<u8>>>>,
}

pub struct RowPackDecodeScratch {
    pub string: String,
    pub bytes: Vec<u8>,
    pub float32: Vec<f32>,
    pub float64: Vec<f64>,
    pub tags: Vec<u8>,
    pub bools: Vec<bool>,
}

impl Default for RowPackDecodeScratch {
    fn default() -> Self {
        Self {
            string: String::with_capacity(1024),
            bytes: Vec::with_capacity(1024),
            float32: Vec::with_capacity(1024),
            float64: Vec::with_capacity(1024),
            tags: Vec::with_capacity(64),
            bools: Vec::with_capacity(64),
        }
    }
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

    pub fn decode_row_to_builders(
        &self,
        row: &[u8],
        builders: &mut [Box<dyn arrow_array::builder::ArrayBuilder>],
        requested_mask: &[bool],
        scratch: &mut RowPackDecodeScratch,
    ) -> Result<()> {
        use arrow_array::builder::BooleanBuilder;
        let mut reader = BitReader::new(row);
        let col_count = self.ctx.columns.len();

        // 1. Tags
        scratch.tags.clear();
        for _ in 0..col_count {
            scratch.tags.push(reader.read_bits(2)?);
        }
        reader.align_to_byte();

        // 2. Bools
        scratch.bools.clear();
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            if matches!(col.encoding, ColumnEncoding::Bool) && scratch.tags[idx] != 0 {
                scratch.bools.push(reader.read_bits(1)? != 0);
            }
        }
        reader.align_to_byte();

        // 3. Values
        let mut data = &row[row.len() - reader.remaining_bytes()..];
        let mut bool_cursor = 0;
        let mut builder_cursor = 0;

        for (idx, col) in self.ctx.columns.iter().enumerate() {
            let tag = scratch.tags[idx];
            let requested = requested_mask[idx];

            if tag == 0 {
                if requested {
                    append_null(builders[builder_cursor].as_mut());
                    builder_cursor += 1;
                }
                continue;
            }

            if matches!(col.encoding, ColumnEncoding::Bool) {
                let val = scratch.bools[bool_cursor];
                bool_cursor += 1;
                if requested {
                    let b = builders[builder_cursor]
                        .as_any_mut()
                        .downcast_mut::<BooleanBuilder>()
                        .unwrap();
                    b.append_value(val);
                    builder_cursor += 1;
                }
                continue;
            }

            if requested {
                decode_and_append_to_builder(
                    builders[builder_cursor].as_mut(),
                    col,
                    tag,
                    &mut data,
                    self.dict_values[idx].as_ref(),
                    scratch,
                )?;
                builder_cursor += 1;
            } else {
                decode_value_skip(col, tag, &mut data, self.dict_values[idx].as_ref())?;
            }
        }
        Ok(())
    }

    pub fn decode_batch_to_arrow<T: AsRef<[u8]>>(
        &self,
        rows: &[T],
        col_indices: &[usize],
    ) -> Result<arrow_array::RecordBatch> {
        use arrow_array::builder::*;
        use arrow_schema::Schema;
        use std::sync::Arc;

        let mut builders = Vec::with_capacity(col_indices.len());
        let mut fields = Vec::with_capacity(col_indices.len());

        for &col_idx in col_indices {
            if col_idx >= self.ctx.columns.len() {
                return Err(anyhow!("column index out of bounds"));
            }
            let col = &self.ctx.columns[col_idx];
            let (field, builder) = make_arrow_builder(&col.name, col.logical_type, rows.len())?;
            fields.push(field);
            builders.push(builder);
        }

        let mut requested_mask = vec![false; self.ctx.columns.len()];
        for &idx in col_indices {
            requested_mask[idx] = true;
        }

        let mut scratch = RowPackDecodeScratch::default();

        for row in rows {
            self.decode_row_to_builders(
                row.as_ref(),
                &mut builders,
                &requested_mask,
                &mut scratch,
            )?;
        }

        let arrays: Vec<Arc<dyn arrow_array::Array>> =
            builders.into_iter().map(|mut b| b.finish()).collect();
        let schema = Schema::new(fields);
        RecordBatch::try_new(Arc::new(schema), arrays)
            .map_err(|e: arrow_schema::ArrowError| anyhow!(e))
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
            mp::write_str(out, text)?;
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
            mp::write_str(out, text)?;
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
                let byte_len = total_bits.div_ceil(8);
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
                let byte_len = total_bits.div_ceil(8);
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
        if data.is_empty() {
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
        if data.is_empty() {
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

pub(crate) fn format_decimal_from_le(bytes: &[u8], scale: i32) -> String {
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

pub(crate) fn apply_decimal_scale_i128(value: i128, scale: i32) -> i128 {
    if scale == 0 {
        return value;
    }
    if scale > 0 {
        let pow = pow10_i128(scale as u32);
        return value / pow;
    }
    let pow = pow10_i128((-scale) as u32);
    value.saturating_mul(pow)
}

pub(crate) fn apply_decimal_scale_i256(value: arrow_buffer::i256, scale: i32) -> BigInt {
    let mut out = BigInt::from_signed_bytes_le(&value.to_le_bytes());
    if scale == 0 {
        return out;
    }
    if scale > 0 {
        let pow = BigInt::from(10u32).pow(scale as u32);
        out /= pow;
        return out;
    }
    let pow = BigInt::from(10u32).pow((-scale) as u32);
    out *= pow;
    out
}

pub(crate) fn decimal_i128_to_f64(value: i128, scale: i32) -> f64 {
    if scale == 0 {
        return value as f64;
    }
    if scale > 0 {
        return (value as f64) / 10_f64.powi(scale);
    }
    (value as f64) * 10_f64.powi(-scale)
}

pub(crate) fn decimal_i256_to_f64(value: arrow_buffer::i256, scale: i32) -> Option<f64> {
    let out = BigInt::from_signed_bytes_le(&value.to_le_bytes());
    let mut v = out.to_f64()?;
    if scale == 0 {
        return Some(v);
    }
    if scale > 0 {
        v /= 10_f64.powi(scale);
    } else {
        v *= 10_f64.powi(-scale);
    }
    Some(v)
}

fn pow10_i128(exp: u32) -> i128 {
    let mut out = 1i128;
    for _ in 0..exp {
        out = out.saturating_mul(10);
    }
    out
}

pub(crate) fn format_decimal_i128(value: i128, scale: i32) -> String {
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

pub(crate) fn format_decimal_i256(value: arrow_buffer::i256, scale: i32) -> String {
    let out = BigInt::from_signed_bytes_le(&value.to_le_bytes());
    let negative = out.sign() == num_bigint::Sign::Minus;
    let mut digits = out.abs().to_string();

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
            let byte_len = total_bits.div_ceil(8);
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
            let byte_len = total_bits.div_ceil(8);
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
            let byte_len = total_bits.div_ceil(8);
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
            let byte_len = total_bits.div_ceil(8);
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

fn append_null(builder: &mut dyn arrow_array::builder::ArrayBuilder) {
    use arrow_array::builder::*;
    if let Some(b) = builder.as_any_mut().downcast_mut::<BooleanBuilder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder.as_any_mut().downcast_mut::<Int8Builder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder.as_any_mut().downcast_mut::<Int16Builder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder.as_any_mut().downcast_mut::<Int32Builder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder.as_any_mut().downcast_mut::<Int64Builder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder.as_any_mut().downcast_mut::<UInt8Builder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder.as_any_mut().downcast_mut::<UInt16Builder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder.as_any_mut().downcast_mut::<UInt32Builder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder.as_any_mut().downcast_mut::<UInt64Builder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder.as_any_mut().downcast_mut::<Float32Builder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder.as_any_mut().downcast_mut::<Float64Builder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder.as_any_mut().downcast_mut::<StringBuilder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder.as_any_mut().downcast_mut::<BinaryBuilder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder
        .as_any_mut()
        .downcast_mut::<TimestampMicrosecondBuilder>()
    {
        b.append_null();
        return;
    }
    if let Some(b) = builder.as_any_mut().downcast_mut::<Date32Builder>() {
        b.append_null();
        return;
    }
    if let Some(b) = builder
        .as_any_mut()
        .downcast_mut::<ListBuilder<Float32Builder>>()
    {
        b.append_null();
        return;
    }
    if let Some(b) = builder
        .as_any_mut()
        .downcast_mut::<ListBuilder<Float64Builder>>()
    {
        b.append_null();
    }
}

pub fn make_arrow_builder(
    name: &str,
    dtype: IngestedType,
    capacity: usize,
) -> Result<(
    arrow_schema::Field,
    Box<dyn arrow_array::builder::ArrayBuilder>,
)> {
    use arrow_array::builder::*;
    use arrow_schema::{DataType, TimeUnit};

    let (arrow_type, builder): (DataType, Box<dyn ArrayBuilder>) = match dtype {
        IngestedType::Bool => (
            DataType::Boolean,
            Box::new(BooleanBuilder::with_capacity(capacity)),
        ),
        IngestedType::Int8 => (
            DataType::Int8,
            Box::new(Int8Builder::with_capacity(capacity)),
        ),
        IngestedType::Int16 => (
            DataType::Int16,
            Box::new(Int16Builder::with_capacity(capacity)),
        ),
        IngestedType::Int32 => (
            DataType::Int32,
            Box::new(Int32Builder::with_capacity(capacity)),
        ),
        IngestedType::Int64 => (
            DataType::Int64,
            Box::new(Int64Builder::with_capacity(capacity)),
        ),
        IngestedType::UInt8 => (
            DataType::UInt8,
            Box::new(UInt8Builder::with_capacity(capacity)),
        ),
        IngestedType::UInt16 => (
            DataType::UInt16,
            Box::new(UInt16Builder::with_capacity(capacity)),
        ),
        IngestedType::UInt32 => (
            DataType::UInt32,
            Box::new(UInt32Builder::with_capacity(capacity)),
        ),
        IngestedType::UInt64 => (
            DataType::UInt64,
            Box::new(UInt64Builder::with_capacity(capacity)),
        ),
        IngestedType::Float32 => (
            DataType::Float32,
            Box::new(Float32Builder::with_capacity(capacity)),
        ),
        IngestedType::Float64 => (
            DataType::Float64,
            Box::new(Float64Builder::with_capacity(capacity)),
        ),
        IngestedType::String
        | IngestedType::Uuid
        | IngestedType::Uuid256
        | IngestedType::Ip
        | IngestedType::Decimal128
        | IngestedType::Decimal256 => (
            DataType::Utf8,
            Box::new(StringBuilder::with_capacity(capacity, capacity * 32)),
        ),
        IngestedType::Bytes => (
            DataType::Binary,
            Box::new(BinaryBuilder::with_capacity(capacity, capacity * 32)),
        ),
        IngestedType::Timestamp => (
            DataType::Timestamp(TimeUnit::Microsecond, None),
            Box::new(TimestampMicrosecondBuilder::with_capacity(capacity)),
        ),
        IngestedType::Date => (
            DataType::Date32,
            Box::new(Date32Builder::with_capacity(capacity)),
        ),
        IngestedType::Float32Array | IngestedType::Float16Array => {
            let values_builder = Float32Builder::with_capacity(capacity * 32);
            (
                DataType::new_list(DataType::Float32, true),
                Box::new(ListBuilder::new(values_builder)),
            )
        }
        IngestedType::Float64Array => {
            let values_builder = Float64Builder::with_capacity(capacity * 32);
            (
                DataType::new_list(DataType::Float64, true),
                Box::new(ListBuilder::new(values_builder)),
            )
        }
    };

    Ok((arrow_schema::Field::new(name, arrow_type, true), builder))
}

fn decode_and_append_to_builder(
    builder: &mut dyn arrow_array::builder::ArrayBuilder,
    col: &ColumnContext,
    tag: u8,
    data: &mut &[u8],
    dict: Option<&Vec<Vec<u8>>>,
    scratch: &mut RowPackDecodeScratch,
) -> Result<()> {
    use arrow_array::builder::*;

    // Helper to get typed builder. We trust the builder matches the column type because we created it that way.
    // However, we should be careful.

    match &col.encoding {
        ColumnEncoding::Int {
            signed,
            varint,
            bits,
        } => {
            if *signed {
                let v = decode_int(data, *varint, *bits)?;
                match col.logical_type {
                    IngestedType::Int64 => builder
                        .as_any_mut()
                        .downcast_mut::<Int64Builder>()
                        .unwrap()
                        .append_value(v),
                    IngestedType::Int32 => builder
                        .as_any_mut()
                        .downcast_mut::<Int32Builder>()
                        .unwrap()
                        .append_value(v as i32),
                    IngestedType::Int16 => builder
                        .as_any_mut()
                        .downcast_mut::<Int16Builder>()
                        .unwrap()
                        .append_value(v as i16),
                    IngestedType::Int8 => builder
                        .as_any_mut()
                        .downcast_mut::<Int8Builder>()
                        .unwrap()
                        .append_value(v as i8),
                    _ => return Err(anyhow!("rowpack int type mismatch")),
                }
            } else {
                let v = decode_uint(data, *varint, *bits)?;
                match col.logical_type {
                    IngestedType::UInt64 => builder
                        .as_any_mut()
                        .downcast_mut::<UInt64Builder>()
                        .unwrap()
                        .append_value(v),
                    IngestedType::UInt32 => builder
                        .as_any_mut()
                        .downcast_mut::<UInt32Builder>()
                        .unwrap()
                        .append_value(v as u32),
                    IngestedType::UInt16 => builder
                        .as_any_mut()
                        .downcast_mut::<UInt16Builder>()
                        .unwrap()
                        .append_value(v as u16),
                    IngestedType::UInt8 => builder
                        .as_any_mut()
                        .downcast_mut::<UInt8Builder>()
                        .unwrap()
                        .append_value(v as u8),
                    _ => return Err(anyhow!("rowpack uint type mismatch")),
                }
            }
        }
        ColumnEncoding::Float { storage } => {
            let v = decode_float(data, *storage)?;
            match col.logical_type {
                IngestedType::Float32 => builder
                    .as_any_mut()
                    .downcast_mut::<Float32Builder>()
                    .unwrap()
                    .append_value(v as f32),
                IngestedType::Float64 => builder
                    .as_any_mut()
                    .downcast_mut::<Float64Builder>()
                    .unwrap()
                    .append_value(v),
                _ => return Err(anyhow!("rowpack float type mismatch")),
            }
        }
        ColumnEncoding::Bytes { fixed_len } => {
            let bytes = if tag == 2 {
                decode_bytes(data, None)?
            } else {
                decode_bytes(data, *fixed_len)?
            };
            builder
                .as_any_mut()
                .downcast_mut::<BinaryBuilder>()
                .unwrap()
                .append_value(bytes);
        }
        ColumnEncoding::Uuid => {
            let bytes = decode_fixed_bytes::<16>(data)?;
            let mut buf = [0u8; 36];
            let s = uuid_simd::format_hyphenated(
                &bytes,
                uuid_simd::AsOut::as_out(&mut buf),
                uuid_simd::AsciiCase::Lower,
            );
            let s = std::str::from_utf8(s).unwrap();
            builder
                .as_any_mut()
                .downcast_mut::<StringBuilder>()
                .unwrap()
                .append_value(s);
        }
        ColumnEncoding::Uuid256 => {
            let bytes = decode_fixed_bytes::<32>(data)?;
            let mut buf = [0u8; 64];
            let bytes_ptr = bytes.as_ptr();
            let hi = unsafe { &*(bytes_ptr as *const [u8; 16]) };
            let lo = unsafe { &*(bytes_ptr.add(16) as *const [u8; 16]) };
            let out_ptr = buf.as_mut_ptr();
            let hi_out = unsafe { &mut *(out_ptr as *mut [u8; 32]) };
            let lo_out = unsafe { &mut *(out_ptr.add(32) as *mut [u8; 32]) };
            let _ = uuid_simd::format_simple(
                hi,
                uuid_simd::AsOut::as_out(hi_out),
                uuid_simd::AsciiCase::Lower,
            );
            let _ = uuid_simd::format_simple(
                lo,
                uuid_simd::AsOut::as_out(lo_out),
                uuid_simd::AsciiCase::Lower,
            );
            let s = std::str::from_utf8(&buf).unwrap();
            builder
                .as_any_mut()
                .downcast_mut::<StringBuilder>()
                .unwrap()
                .append_value(s);
        }
        ColumnEncoding::Ip => {
            let bytes = decode_fixed_bytes::<16>(data)?;
            let s = format_ip(bytes);
            builder
                .as_any_mut()
                .downcast_mut::<StringBuilder>()
                .unwrap()
                .append_value(s);
        }
        ColumnEncoding::Date { varint, bits } => {
            let v = decode_int(data, *varint, *bits)?;
            builder
                .as_any_mut()
                .downcast_mut::<Date32Builder>()
                .unwrap()
                .append_value(v as i32);
        }
        ColumnEncoding::Timestamp => {
            let secs = zigzag_decode(read_varint(data)?);
            let nanos = read_varint(data)? as u32;
            let micros = secs
                .saturating_mul(1_000_000)
                .saturating_add(nanos as i64 / 1000);
            builder
                .as_any_mut()
                .downcast_mut::<TimestampMicrosecondBuilder>()
                .unwrap()
                .append_value(micros);
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
            builder
                .as_any_mut()
                .downcast_mut::<StringBuilder>()
                .unwrap()
                .append_value(text);
        }
        ColumnEncoding::String { encoding } => {
            // We use scratch string to decode, then append to builder.
            // This is safe because builder copies the string immediately.
            scratch.string.clear();
            if tag == 2 {
                decode_string_fallback_into(data, &mut scratch.string)?;
            } else {
                decode_string_into(data, encoding, dict, scratch)?;
            }
            builder
                .as_any_mut()
                .downcast_mut::<StringBuilder>()
                .unwrap()
                .append_value(&scratch.string);
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
                    let b = builder
                        .as_any_mut()
                        .downcast_mut::<ListBuilder<Float32Builder>>()
                        .unwrap();
                    b.values().append_slice(&scratch.float32);
                    b.append(true);
                }
                IngestedType::Float64Array => {
                    scratch.float64.clear();
                    for _ in 0..len {
                        let v = decode_float(data, *storage)?;
                        scratch.float64.push(v);
                    }
                    let b = builder
                        .as_any_mut()
                        .downcast_mut::<ListBuilder<Float64Builder>>()
                        .unwrap();
                    b.values().append_slice(&scratch.float64);
                    b.append(true);
                }
                _ => return Err(anyhow!("rowpack invalid array type")),
            }
        }
        ColumnEncoding::Bool => return Err(anyhow!("rowpack bool unexpected")),
    }
    Ok(())
}
