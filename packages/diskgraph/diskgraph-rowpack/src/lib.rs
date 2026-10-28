// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use arrow_array::{Array, RecordBatch};
use base64::Engine;
use chrono::{DateTime, Utc};
use diskgraph_core::time::{convert_trunc, split_parts, TimeUnit};
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
mod tests_int_string_list;
#[cfg(test)]
mod tests_invalid_as_null;
#[cfg(test)]
mod tests_string_float_array;
#[cfg(test)]
mod tests_type_matrix;
#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests_unit;

pub use crate::access::{
    BinaryStringHint, FloatListValues, FloatListView, IntListValues, IntListView, RowAccess,
    RowAccessHints, StringListValues, StringListView,
};
pub use crate::parser::{detect_encoded_int_format, detect_timestamp_format, TIMESTAMP_FORMATS};

pub(crate) const ROWPACK_VERSION: u8 = 3;
pub(crate) const MAX_INPUT_STRING_BYTES: usize = 64 * 1024 * 1024;
const MAX_FIELD_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const MAX_ARRAY_LEN: usize = 1_000_000;
const ALPHABET_MAX_CHARS: usize = 64;
const ALPHABET_MIN_COVERAGE: f32 = 0.95;
const ALPHABET_INVALID: u8 = u8::MAX;
/// Maximum unique values tracked per column during context-phase sampling.
/// Matches the string_counts cap. Beyond this, stype is resolvable without exact count.
const MAX_SAMPLE_UNIQUES: usize = 1_000_000;
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
            sample_rows: Some(100_000),
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
        self.sample_rows.unwrap_or(100_000)
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
    Int8Array,
    Int16Array,
    Int32Array,
    Int64Array,
    StringArray,
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
    Int8Array,
    Int16Array,
    Int32Array,
    Int64Array,
    StringArray {
        encoding: StringEncoding,
    },
    FixedFloatArray {
        storage: FloatStorage,
        len: usize,
    },
    FixedIntArray {
        len: usize,
    },
    /// Tag 1: varint index into frequency-sorted `values`.
    /// Tag 2: miss, written with `varint`/`bits` so it is break-even with plain.
    IntDict {
        values: Vec<i64>,
        signed: bool,
        varint: bool,
        bits: u8,
    },
    /// Tag 1: varint index into frequency-sorted `values`.
    /// Tag 2: miss, written at `storage` width so it is break-even with plain.
    FloatDict {
        values: Vec<f64>,
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

#[derive(Debug, Clone, PartialEq)]
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
    /// Ticks of the index clock.
    Timestamp(i64),
    Decimal128(i128),
    Decimal256([u8; 32]),
    Float32Array(&'a [f32]),
    Float64Array(&'a [f64]),
    Int8Array(&'a [i8]),
    Int16Array(&'a [i16]),
    Int32Array(&'a [i32]),
    Int64Array(&'a [i64]),
    StringArray(&'a [&'a str]),
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
                IngestedType::Bool => match access_or_null(access.get_bool(idx)) {
                    None => stats.observe_null(),
                    Some(_) => stats.observe_bool(),
                },
                IngestedType::Int64 => match access_or_null(access.get_int64(idx)) {
                    None => stats.observe_null(),
                    Some(v) => stats.observe_int64(v),
                },
                IngestedType::Int16 | IngestedType::Int8 | IngestedType::Int32 => {
                    match access_or_null(access.get_int32(idx)) {
                        None => stats.observe_null(),
                        Some(v) => stats.observe_int32(v),
                    }
                }
                IngestedType::UInt64 => match access_or_null(access.get_uint64(idx)) {
                    None => stats.observe_null(),
                    Some(v) => stats.observe_uint64(v),
                },
                IngestedType::UInt16 | IngestedType::UInt8 | IngestedType::UInt32 => {
                    match access_or_null(access.get_uint32(idx)) {
                        None => stats.observe_null(),
                        Some(v) => stats.observe_uint32(v),
                    }
                }
                IngestedType::Float64 => match access_or_null(access.get_float64(idx)) {
                    None => stats.observe_null(),
                    Some(v) => stats.observe_float64(v),
                },
                IngestedType::Float32 => match access_or_null(access.get_float32(idx)) {
                    None => stats.observe_null(),
                    Some(v) => stats.observe_float32(v),
                },
                IngestedType::String => {
                    match access.with_string(idx, |v| {
                        stats.observe_string(v, &self.config);
                        Ok(())
                    }) {
                        Ok(Some(_)) => {}
                        Ok(None) | Err(_) => stats.observe_null(),
                    }
                }
                IngestedType::Bytes => match access.with_bytes(idx, |v| Ok(v.len())) {
                    Ok(None) | Err(_) => stats.observe_null(),
                    Ok(Some(len)) => stats.observe_bytes_len(len),
                },
                IngestedType::Uuid => match access_or_null(access.get_uuid(idx)) {
                    None => stats.observe_null(),
                    Some(_) => stats.observe_uuid(),
                },
                IngestedType::Uuid256 => match access_or_null(access.get_uuid256(idx)) {
                    None => stats.observe_null(),
                    Some(_) => stats.observe_uuid256(),
                },
                IngestedType::Ip => match access_or_null(access.get_ip(idx)) {
                    None => stats.observe_null(),
                    Some(_) => stats.observe_ip(),
                },
                IngestedType::Date => match access_or_null(access.get_date(idx)) {
                    None => stats.observe_null(),
                    Some(_) => stats.count = stats.count.saturating_add(1),
                },
                IngestedType::Timestamp => match access_or_null(access.get_timestamp(idx)) {
                    None => stats.observe_null(),
                    Some(secs) => stats.observe_timestamp(secs),
                },
                IngestedType::Decimal128 => match access_or_null(access.get_decimal128(idx)) {
                    None => stats.observe_null(),
                    Some(_) => stats.count = stats.count.saturating_add(1),
                },
                IngestedType::Decimal256 => match access_or_null(access.get_decimal256(idx)) {
                    None => stats.observe_null(),
                    Some(_) => stats.count = stats.count.saturating_add(1),
                },
                IngestedType::Float32Array | IngestedType::Float16Array => {
                    match access.with_float32_array(idx, |values| {
                        stats.observe_array_f32(values);
                        Ok(())
                    }) {
                        Ok(Some(_)) => {}
                        Ok(None) | Err(_) => stats.observe_null(),
                    }
                }
                IngestedType::Float64Array => {
                    match access.with_float64_array(idx, |values| {
                        stats.observe_array_f64(values);
                        Ok(())
                    }) {
                        Ok(Some(_)) => {}
                        Ok(None) | Err(_) => stats.observe_null(),
                    }
                }
                IngestedType::Int8Array
                | IngestedType::Int16Array
                | IngestedType::Int32Array
                | IngestedType::Int64Array => {
                    // `with_int64_array` widens narrower i8/i16/i32 list
                    // elements to i64, so the stats path is shared across
                    // every int-array width.
                    match access.with_int64_array(idx, |values| {
                        for &v in values {
                            stats.observe_int_stats(v);
                        }
                        Ok(())
                    }) {
                        Ok(Some(_)) => {}
                        Ok(None) | Err(_) => stats.observe_null(),
                    }
                }
                IngestedType::StringArray => {
                    match access.with_string_array(idx, |tokens| {
                        for &token in tokens {
                            stats.observe_string(token, &self.config);
                        }
                        Ok(())
                    }) {
                        Ok(Some(_)) => {}
                        Ok(None) | Err(_) => stats.observe_null(),
                    }
                }
            }
        }
        self.samples_seen += 1;
        Ok(())
    }

    /// Returns per-column `(num_unique, count)` from the context-phase sample.
    ///
    /// For integer columns: uses `int_counts.len()` (capped at 1M).
    /// For string columns: uses `string_counts.len()` (capped at 1M).
    /// For other types: `num_unique` is 0 (stype is unambiguous from dtype).
    pub fn sample_unique_counts(&self) -> Vec<(usize, usize)> {
        self.stats
            .iter()
            .zip(self.columns.iter())
            .map(|(s, logical)| {
                let num_unique = match logical {
                    IngestedType::Int8
                    | IngestedType::Int16
                    | IngestedType::Int32
                    | IngestedType::Int64
                    | IngestedType::UInt8
                    | IngestedType::UInt16
                    | IngestedType::UInt32
                    | IngestedType::UInt64 => s.int_counts.len(),
                    IngestedType::String => s.string_counts.len(),
                    _ => 0,
                };
                (num_unique, s.count)
            })
            .collect()
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
        ColumnEncoding::IntDict { values, .. } => {
            // Frequency-sorted, so most rows hit a shorter index than the last.
            varint_bytes_u64(values.len().saturating_sub(1) as u64 / 2)
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
            varint_bytes(secs_bits)
        }
        ColumnEncoding::FloatDict { values, .. } => {
            // Frequency-sorted, so most rows hit a shorter index than the last.
            varint_bytes_u64(values.len().saturating_sub(1) as u64 / 2)
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
        ColumnEncoding::FixedFloatArray { storage, len } => {
            let elem = match storage {
                FloatStorage::F16 => 2,
                FloatStorage::F32 => 4,
                FloatStorage::F64 => 8,
            };
            len * elem
        }
        ColumnEncoding::FixedIntArray { len } => len * 2,
        ColumnEncoding::Int8Array => {
            // varint(N) + N * varint(i8 widened). i8 always fits in 1 varint byte.
            let avg_len = avg_len(stats.array_len_total, stats.array_count);
            varint_bytes_u64(avg_len as u64) + avg_len
        }
        ColumnEncoding::Int16Array => {
            // varint(N) + N * varint(i16 widened). Most i16 values fit in 1-2 bytes.
            let avg_len = avg_len(stats.array_len_total, stats.array_count);
            varint_bytes_u64(avg_len as u64) + avg_len * 2
        }
        ColumnEncoding::Int32Array => {
            // varint(N) + N * varint(i32 widened). Match the i64 assumption.
            let avg_len = avg_len(stats.array_len_total, stats.array_count);
            varint_bytes_u64(avg_len as u64) + avg_len * 2
        }
        ColumnEncoding::Int64Array => {
            // varint(N) + N * varint(i64) — assume average 2 bytes per varint
            let avg_len = avg_len(stats.array_len_total, stats.array_count);
            varint_bytes_u64(avg_len as u64) + avg_len * 2
        }
        ColumnEncoding::StringArray { encoding } => {
            // varint(N) + N * (per-token cost)
            let avg_tokens = avg_len(stats.array_len_total, stats.array_count);
            let token_cost = match encoding {
                StringEncoding::Dict { .. } => varint_bytes_u64(1024),
                StringEncoding::Raw => {
                    let avg = avg_len(stats.string_bytes, stats.string_total);
                    varint_bytes_u64(avg as u64) + avg
                }
                _ => {
                    let avg = avg_len(stats.string_bytes, stats.string_total);
                    varint_bytes_u64(avg as u64) + avg
                }
            };
            varint_bytes_u64(avg_tokens as u64) + avg_tokens * token_cost
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
    total.checked_div(count).unwrap_or(0)
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
    int_counts: FastHashMap<i64, u32>,
    /// Keyed on the bit pattern so every distinct value stays distinct.
    float_counts: FastHashMap<u64, u32>,
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
            int_counts: FastHashMap::new(),
            float_counts: FastHashMap::new(),
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

    /// `varint` is the incumbent decision the dictionary has to beat.
    fn int_encoding(
        &self,
        signed: bool,
        bits: u8,
        varint: bool,
        config: &RowPackConfig,
    ) -> ColumnEncoding {
        let plain = ColumnEncoding::Int {
            signed,
            varint,
            bits,
        };
        let observed: u64 = self.int_counts.values().map(|c| *c as u64).sum();
        if self.int_counts.is_empty() {
            return plain;
        }
        let unique = self.int_counts.len();
        if unique > config.max_dict_entries() {
            return plain;
        }

        // Tie-break by value: keeps the context deterministic across runs.
        let mut entries: Vec<(i64, u32)> = self
            .int_counts
            .iter()
            .map(|(value, count)| (*value, *count))
            .collect();
        entries.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let value_bytes = |v: i64| -> usize {
            if varint {
                varint_bytes_u64(zigzag_encode(v))
            } else {
                bits as usize / 8
            }
        };
        let plain_bytes: u64 = entries
            .iter()
            .map(|(v, c)| value_bytes(*v) as u64 * *c as u64)
            .sum();

        // Cap the dictionary so the widest index costs no more than the cheapest
        // value would have cost plain. Index widths are frozen from the sample
        // but the distribution beyond it is not, so without this a column whose
        // rare values become common later pays a 2-byte index where plain spent
        // 1. Entries past the cap fall back, which is already break-even.
        let cheapest_plain = entries
            .iter()
            .map(|(v, _)| value_bytes(*v))
            .min()
            .unwrap_or(1);
        let max_entries = 1usize << (7 * cheapest_plain.min(8));
        let kept = entries.len().min(max_entries).max(1);

        let mut dict_bytes: u64 = 0;
        for (idx, (v, count)) in entries.iter().enumerate() {
            let per_row = if idx < kept {
                varint_bytes_u64(idx as u64)
            } else {
                value_bytes(*v)
            };
            dict_bytes += per_row as u64 * *count as u64;
        }
        entries.truncate(kept);
        let ctx_bytes: u64 = entries.len() as u64 * 8;

        // Values must repeat, or nearly every row takes the fallback.
        let coverage = observed as f64 / unique as f64;
        if coverage < 2.0 {
            return plain;
        }

        if dict_bytes + ctx_bytes < plain_bytes {
            ColumnEncoding::IntDict {
                values: entries.into_iter().map(|(value, _)| value).collect(),
                signed,
                varint,
                bits,
            }
        } else {
            plain
        }
    }

    /// Floats are picked by width alone -- `choose_float_storage` takes only
    /// whether the source was f64 and never looks at the data. A column of 212
    /// distinct item counts therefore pays 8 bytes a row. Price a dictionary
    /// against plain storage the same way `int_encoding` does.
    fn float_encoding(&self, storage: FloatStorage, config: &RowPackConfig) -> ColumnEncoding {
        let plain = ColumnEncoding::Float { storage };
        if self.float_counts.is_empty() {
            return plain;
        }
        let observed: u64 = self.float_counts.values().map(|c| *c as u64).sum();
        let unique = self.float_counts.len();
        if unique > config.max_dict_entries() {
            return plain;
        }
        let width = match storage {
            // A 3-byte index can exceed a 2-byte value, so F16 cannot hold the
            // never-worse-than-plain bound the cap-free cost model relies on.
            // No scalar float selects F16 today; this keeps that assumption
            // from silently breaking if one ever does.
            FloatStorage::F16 => return plain,
            FloatStorage::F32 => 4usize,
            FloatStorage::F64 => 8,
        };

        // Tie-break by value: keeps the context deterministic across runs.
        let mut entries: Vec<(u64, u32)> = self
            .float_counts
            .iter()
            .map(|(bits, count)| (*bits, *count))
            .collect();
        entries.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // No index cap here, unlike the integer path: a plain float is 4 or 8
        // bytes, so a varint index would need 2^28 entries to cost more, and
        // `max_dict_entries` binds long before that.
        let mut dict_bytes = 0u64;
        for (idx, (_, count)) in entries.iter().enumerate() {
            dict_bytes += varint_bytes_u64(idx as u64) as u64 * *count as u64;
        }
        let ctx_bytes = entries.len() as u64 * 8;

        // Values must repeat, or nearly every row takes the fallback.
        if (observed as f64 / unique as f64) < 2.0 {
            return plain;
        }
        if dict_bytes + ctx_bytes < observed * width as u64 {
            ColumnEncoding::FloatDict {
                values: entries
                    .into_iter()
                    .map(|(bits, _)| f64::from_bits(bits))
                    .collect(),
                storage,
            }
        } else {
            plain
        }
    }

    fn observe_int_key(&mut self, v: i64) {
        match self.int_counts.get_mut(&v) {
            Some(count) => *count = count.saturating_add(1),
            None => {
                if self.int_counts.len() < MAX_SAMPLE_UNIQUES {
                    self.int_counts.insert(v, 1);
                }
            }
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
                self.observe_int_key(*v);
            }
            ValueRef::Int32(v) => {
                let v = *v as i64;
                self.int_min = self.int_min.min(v);
                self.int_max = self.int_max.max(v);
                self.observe_int_key(v);
            }
            ValueRef::UInt64(v) => {
                self.uint_max = self.uint_max.max(*v);
                self.observe_int_key(*v as i64);
            }
            ValueRef::UInt32(v) => {
                self.uint_max = self.uint_max.max(*v as u64);
                self.observe_int_key(*v as i64);
            }
            ValueRef::Float64(v) => self.observe_scalar_float(*v),
            ValueRef::Float32(v) => self.observe_scalar_float(*v as f64),
            ValueRef::Bool(_) => {
                self.bool_seen = true;
            }
            ValueRef::String(v) => {
                self.string_total += 1;
                let bytes = v.as_bytes();
                self.string_max_len = self.string_max_len.max(bytes.len());
                self.string_bytes += bytes.len();
                if self.string_counts.len() < MAX_SAMPLE_UNIQUES {
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
            ValueRef::Timestamp(secs) => {
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
            ValueRef::Int8Array(values) => {
                self.array_len_total += values.len();
                self.array_count += 1;
                for &v in *values {
                    self.observe_int_stats(v as i64);
                }
            }
            ValueRef::Int16Array(values) => {
                self.array_len_total += values.len();
                self.array_count += 1;
                for &v in *values {
                    self.observe_int_stats(v as i64);
                }
            }
            ValueRef::Int32Array(values) => {
                self.array_len_total += values.len();
                self.array_count += 1;
                for &v in *values {
                    self.observe_int_stats(v as i64);
                }
            }
            ValueRef::Int64Array(values) => {
                self.array_len_total += values.len();
                self.array_count += 1;
                for &v in *values {
                    self.observe_int_stats(v);
                }
            }
            ValueRef::StringArray(tokens) => {
                for &token in *tokens {
                    self.observe_string(token, config);
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

    /// Range stats only. Array elements come here: an int-array column
    /// resolves to `Int64Array`/`FixedIntArray` and never reaches
    /// `int_encoding`, so recording a dictionary key per element cannot change
    /// any decision. The range is still needed, since the array encoding picks
    /// varint by magnitude.
    fn observe_int_stats(&mut self, v: i64) {
        self.count += 1;
        self.int_min = self.int_min.min(v);
        self.int_max = self.int_max.max(v);
    }

    fn observe_int64(&mut self, v: i64) {
        self.observe_int_stats(v);
        self.observe_int_key(v);
    }

    fn observe_int32(&mut self, v: i32) {
        self.observe_int64(v as i64);
    }

    fn observe_uint64(&mut self, v: u64) {
        self.count += 1;
        self.uint_max = self.uint_max.max(v);
        self.observe_int_key(v as i64);
    }

    fn observe_uint32(&mut self, v: u32) {
        self.observe_uint64(v as u64);
    }

    fn observe_float64(&mut self, v: f64) {
        self.count += 1;
        self.observe_scalar_float(v);
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
        if self.string_counts.len() < MAX_SAMPLE_UNIQUES {
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

    /// Records a scalar float for the dictionary. Keyed on bits, so -0.0 and
    /// 0.0 stay distinct. Deliberately not called for array elements: an array
    /// column resolves to `FloatArray`/`FixedFloatArray` and never reaches
    /// `float_encoding`, so recording its elements cannot change any decision,
    /// and it costs a hash insert per element rather than per row.
    fn observe_float_key(&mut self, v: f64) {
        let key = v.to_bits();
        match self.float_counts.get_mut(&key) {
            Some(count) => *count = count.saturating_add(1),
            None => {
                if self.float_counts.len() < MAX_SAMPLE_UNIQUES {
                    self.float_counts.insert(key, 1);
                }
            }
        }
    }

    /// Scalar float observation: distribution stats plus the dictionary key.
    /// One entry point so the two cannot drift apart, which is how the encoder
    /// and the dictionary ended up seeing different value sets before.
    fn observe_scalar_float(&mut self, v: f64) {
        self.observe_float_stats(v);
        self.observe_float_key(v);
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
            IngestedType::Int64 => {
                Ok(self.int_encoding(true, 64, int_use_varint(self.int_min, self.int_max), config))
            }
            IngestedType::Int32 => {
                Ok(self.int_encoding(true, 32, int_use_varint(self.int_min, self.int_max), config))
            }
            IngestedType::Int16 => {
                Ok(self.int_encoding(true, 16, int_use_varint(self.int_min, self.int_max), config))
            }
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
            IngestedType::Float64 => Ok(self.float_encoding(choose_float_storage(true), config)),
            IngestedType::Float32 => Ok(self.float_encoding(choose_float_storage(false), config)),
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
            IngestedType::Float32Array => Ok(match fixed_len {
                Some(len) => ColumnEncoding::FixedFloatArray {
                    storage: choose_float_storage(false),
                    len,
                },
                None => ColumnEncoding::FloatArray {
                    storage: choose_float_storage(false),
                },
            }),
            IngestedType::Float64Array => Ok(match fixed_len {
                Some(len) => ColumnEncoding::FixedFloatArray {
                    storage: choose_float_storage(true),
                    len,
                },
                None => ColumnEncoding::FloatArray {
                    storage: choose_float_storage(true),
                },
            }),
            IngestedType::Float16Array => Ok(match fixed_len {
                Some(len) => ColumnEncoding::FixedFloatArray {
                    storage: FloatStorage::F16,
                    len,
                },
                None => ColumnEncoding::FloatArray {
                    storage: FloatStorage::F16,
                },
            }),
            IngestedType::Int8Array => Ok(match fixed_len {
                Some(len) => ColumnEncoding::FixedIntArray { len },
                None => ColumnEncoding::Int8Array,
            }),
            IngestedType::Int16Array => Ok(match fixed_len {
                Some(len) => ColumnEncoding::FixedIntArray { len },
                None => ColumnEncoding::Int16Array,
            }),
            IngestedType::Int32Array => Ok(match fixed_len {
                Some(len) => ColumnEncoding::FixedIntArray { len },
                None => ColumnEncoding::Int32Array,
            }),
            IngestedType::Int64Array => Ok(match fixed_len {
                Some(len) => ColumnEncoding::FixedIntArray { len },
                None => ColumnEncoding::Int64Array,
            }),
            IngestedType::StringArray => Ok(ColumnEncoding::StringArray {
                encoding: choose_string_encoding(self, config)?,
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

/// Eligibility alone is not a reason to pick an encoding: alphabet only shrinks
/// a value by `log2(charset)/8` per character, while a dictionary replaces the
/// whole value with a varint index, so a low-cardinality column from a small
/// charset is far cheaper as a dictionary even though both apply.
fn choose_string_encoding(stats: &ColumnStats, config: &RowPackConfig) -> Result<StringEncoding> {
    let total = stats.string_total.max(1) as f32;
    let avg_len = stats
        .string_bytes
        .checked_div(stats.string_total)
        .unwrap_or(0);
    let samples = &stats.string_samples;

    // Pushed in the historical preference order, which is what ties and the
    // no-sample fallback below resolve to.
    let mut candidates: Vec<StringEncoding> = Vec::new();

    if stats.alphabet_ok
        && stats.alphabet_len > 1
        && stats.alphabet_hits as f32 / total >= ALPHABET_MIN_COVERAGE
    {
        let mut alphabet = Vec::with_capacity(stats.alphabet_len);
        for (idx, seen) in stats.alphabet_bytes.iter().enumerate() {
            if *seen {
                alphabet.push(idx as u8);
            }
        }
        candidates.push(StringEncoding::Alphabet { alphabet });
    }

    if let Some(values) = string_dict_values(stats, config, total) {
        candidates.push(StringEncoding::Dict { values });
    }

    if stats.string_long_bytes > 0
        && stats.string_long_compressed as f32 / stats.string_long_raw as f32
            <= config.string_long_min_ratio()
    {
        candidates.push(StringEncoding::Lz4);
    }

    candidates.push(StringEncoding::Raw);

    if samples.is_empty() {
        return Ok(candidates.swap_remove(0));
    }
    let (best_bytes, best) = candidates
        .into_iter()
        .map(|enc| (estimate_string_bytes(samples, &enc), enc))
        .min_by_key(|(bytes, _)| *bytes)
        .expect("raw is always a candidate");

    // BPE carries a trained dictionary and a slower decode, so it has to be
    // meaningfully cheaper than the best alternative rather than merely cheaper.
    if avg_len >= config.bpe_min_avg_len() && avg_len < config.string_long_min_len() {
        let dict = train_bpe_dict(
            samples,
            config.bpe_max_dict_entries(),
            config.bpe_max_dict_bytes(),
        );
        if !dict.is_empty() {
            let bpe_bytes = estimate_bpe_bytes(samples, &dict);
            if bpe_bytes < (best_bytes as f64 * config.bpe_min_gain_ratio() as f64) as u64 {
                return Ok(StringEncoding::Bpe { dict });
            }
        }
    }
    Ok(best)
}

/// The most frequent values that fit the dictionary budget, or `None` when a
/// dictionary would not pay for itself. `avg_hits` is what bounds the context:
/// the *average* entry is repeated at least twice, so the dictionary as a whole
/// holds its values for less than they would cost inline. That is why the cost
/// model prices payloads only.
fn string_dict_values(
    stats: &ColumnStats,
    config: &RowPackConfig,
    total: f32,
) -> Option<Vec<Vec<u8>>> {
    if stats.string_counts.is_empty() {
        return None;
    }
    let mut entries: Vec<(&Vec<u8>, usize)> = stats
        .string_counts
        .iter()
        .map(|(k, v)| (k, *v))
        .collect();
    // Break ties on the value itself: `string_counts` is a randomly-seeded hash
    // map, and a stable sort would otherwise leave equally-frequent values in
    // its iteration order. Indices are priced as varints below, so that would
    // make the chosen encoding differ between runs over identical input.
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let mut values = Vec::new();
    let mut bytes = 0usize;
    let mut coverage = 0usize;
    for (value, count) in entries {
        let next_bytes = bytes + value.len();
        if next_bytes > config.max_dict_bytes() || values.len() >= config.max_dict_entries() {
            break;
        }
        bytes = next_bytes;
        coverage += count;
        values.push(value.clone());
    }
    let avg_hits = coverage as f32 / values.len().max(1) as f32;
    if values.is_empty() || coverage as f32 / total < config.string_dict_coverage() || avg_hits < 2.0
    {
        return None;
    }
    Some(values)
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
    entries.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    entries.into_iter().map(|(bytes, _)| bytes).collect()
}

/// Trait for observing column values during the encode loop.
///
/// When col-stats are **disabled**, `NoOpObserver` is used — LLVM monomorphizes
/// the generic `encode_row_observed` and eliminates all observer calls entirely.
/// When enabled, a real observer (e.g. `FileColStats`) collects streaming statistics.
pub trait ColumnValueObserver {
    fn observe_i64(&mut self, col_idx: usize, value: i64);
    fn observe_f64(&mut self, col_idx: usize, value: f64);
    fn observe_string(&mut self, col_idx: usize, value: &str);
    fn observe_timestamp(&mut self, col_idx: usize, secs: i64);
    fn observe_float_list(&mut self, col_idx: usize, view: &FloatListView);
    fn observe_int64_list(&mut self, col_idx: usize, view: &IntListView);
    fn observe_string_list(&mut self, col_idx: usize, view: &StringListView);
    fn observe_null(&mut self, col_idx: usize);
    fn observe_parse_failure(&mut self, col_idx: usize);
}

/// Zero-sized observer that does nothing. LLVM eliminates all calls via monomorphization.
pub struct NoOpObserver;

impl ColumnValueObserver for NoOpObserver {
    #[inline(always)]
    fn observe_i64(&mut self, _col_idx: usize, _value: i64) {}
    #[inline(always)]
    fn observe_f64(&mut self, _col_idx: usize, _value: f64) {}
    #[inline(always)]
    fn observe_string(&mut self, _col_idx: usize, _value: &str) {}
    #[inline(always)]
    fn observe_timestamp(&mut self, _col_idx: usize, _secs: i64) {}
    #[inline(always)]
    fn observe_float_list(&mut self, _col_idx: usize, _view: &FloatListView) {}
    #[inline(always)]
    fn observe_int64_list(&mut self, _col_idx: usize, _view: &IntListView) {}
    #[inline(always)]
    fn observe_string_list(&mut self, _col_idx: usize, _view: &StringListView) {}
    #[inline(always)]
    fn observe_null(&mut self, _col_idx: usize) {}
    #[inline(always)]
    fn observe_parse_failure(&mut self, _col_idx: usize) {}
}

#[inline(always)]
fn access_or_null<T>(value: Result<Option<T>>) -> Option<T> {
    value.unwrap_or_default()
}

#[inline(always)]
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

pub struct RowPackEncoder {
    ctx: RowPackContext,
    dict_maps: Vec<Option<FastHashMap<Vec<u8>, u32>>>,
    alphabet_maps: Vec<Option<AlphabetMap>>,
    bpe_tries: Vec<Option<BpeTrie>>,
    /// Per-column int-array target bit width, indexed by column position.
    /// `Some(bits)` for `Int{8,16,32,64}Array` columns; `None` otherwise.
    /// Precomputed at construction so the row loop doesn't re-match the
    /// `ColumnEncoding`/`IngestedType` pair on every row.
    int_array_target_bits: Vec<Option<u8>>,
}

#[derive(Debug, Clone, Copy)]
pub struct RowPackRowLens {
    pub tag_bytes: usize,
    pub bool_bytes: usize,
    pub data_bytes: usize,
}

/// Per-column value override used by the FK-free LP "resolution mode"
/// (`ttng-lp-fkfree-resolution-design.md` §S3). For the resolved node-id
/// columns the value is NOT read from the source parquet (it doesn't exist
/// there) — instead the encode worker resolves the PK reference(s) to graph
/// node-id(s) and supplies them here. `encode_row_observed` consults the
/// override at the matching column index and, when present, encodes the
/// supplied value directly INSTEAD of reading from `access`.
///
/// `UInt64` is varint-encoded (not raw bytes), so the scalar override goes
/// through `encode_uint`; the list override is wrapped as an
/// `IntListView { Parsed }` and goes through `encode_int_list_view_with_tag`.
#[derive(Debug, Clone)]
pub enum ResolveOverride {
    /// No override for this column — encode from `access` as usual.
    None,
    /// A resolved scalar node-id for a `UInt64` column.
    Scalar(u64),
    /// Resolved node-ids for an `Int64Array` column (already node-ids; the
    /// `Parsed` view bypasses any Arrow source array).
    List(Vec<i64>),
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
        let mut int_array_target_bits = Vec::with_capacity(ctx.columns.len());
        for col in &ctx.columns {
            int_array_target_bits.push(match col.logical_type {
                IngestedType::Int8Array
                | IngestedType::Int16Array
                | IngestedType::Int32Array
                | IngestedType::Int64Array => {
                    Some(int_array_encoding_bits(&col.encoding, col.logical_type)?)
                }
                _ => None,
            });
            match &col.encoding {
                ColumnEncoding::String {
                    encoding: StringEncoding::Dict { values },
                }
                | ColumnEncoding::StringArray {
                    encoding: StringEncoding::Dict { values },
                } => {
                    let mut map = FastHashMap::with_capacity(values.len());
                    for (idx, value) in values.iter().enumerate() {
                        map.insert(value.clone(), idx as u32);
                    }
                    dict_maps.push(Some(map));
                }
                // LE-byte keys reuse the existing byte-keyed dict plumbing.
                ColumnEncoding::IntDict { values, .. } => {
                    let mut map = FastHashMap::with_capacity(values.len());
                    for (idx, value) in values.iter().enumerate() {
                        map.insert(value.to_le_bytes().to_vec(), idx as u32);
                    }
                    dict_maps.push(Some(map));
                }
                ColumnEncoding::FloatDict { values, .. } => {
                    let mut map = FastHashMap::with_capacity(values.len());
                    for (idx, value) in values.iter().enumerate() {
                        map.insert(value.to_le_bytes().to_vec(), idx as u32);
                    }
                    dict_maps.push(Some(map));
                }
                _ => dict_maps.push(None),
            }
            match &col.encoding {
                ColumnEncoding::String {
                    encoding: StringEncoding::Alphabet { alphabet },
                }
                | ColumnEncoding::StringArray {
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
                }
                | ColumnEncoding::StringArray {
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
            int_array_target_bits,
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
                IngestedType::Bool => match access_or_null(access.get_bool(idx)) {
                    None => 0u8,
                    Some(v) => {
                        scratch.bools.write_bits(if v { 1 } else { 0 }, 1);
                        1u8
                    }
                },
                IngestedType::Int64 => match access_or_null(access.get_int64(idx)) {
                    None => 0u8,
                    Some(v) => match &col.encoding {
                        ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } => {
                            encode_int(&mut scratch.data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        }
                        ColumnEncoding::IntDict { .. } => encode_int_dict(
                            &mut scratch.data,
                            col,
                            self.dict_maps[idx].as_ref(),
                            (v as i128) as i64,
                        )?,
                        _ => return Err(anyhow!("rowpack int64 encoding mismatch")),
                    },
                },
                IngestedType::Int16 => match access_or_null(access.get_int32(idx)) {
                    None => 0u8,
                    Some(v) => match &col.encoding {
                        ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } => {
                            encode_int(&mut scratch.data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        }
                        ColumnEncoding::IntDict { .. } => encode_int_dict(
                            &mut scratch.data,
                            col,
                            self.dict_maps[idx].as_ref(),
                            (v as i128) as i64,
                        )?,
                        _ => return Err(anyhow!("rowpack int16 encoding mismatch")),
                    },
                },
                IngestedType::Int8 => match access_or_null(access.get_int32(idx)) {
                    None => 0u8,
                    Some(v) => match &col.encoding {
                        ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } => {
                            encode_int(&mut scratch.data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        }
                        ColumnEncoding::IntDict { .. } => encode_int_dict(
                            &mut scratch.data,
                            col,
                            self.dict_maps[idx].as_ref(),
                            (v as i128) as i64,
                        )?,
                        _ => return Err(anyhow!("rowpack int8 encoding mismatch")),
                    },
                },
                IngestedType::Int32 => match access_or_null(access.get_int32(idx)) {
                    None => 0u8,
                    Some(v) => match &col.encoding {
                        ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } => {
                            encode_int(&mut scratch.data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        }
                        ColumnEncoding::IntDict { .. } => encode_int_dict(
                            &mut scratch.data,
                            col,
                            self.dict_maps[idx].as_ref(),
                            (v as i128) as i64,
                        )?,
                        _ => return Err(anyhow!("rowpack int32 encoding mismatch")),
                    },
                },
                IngestedType::UInt64 => match access_or_null(access.get_uint64(idx)) {
                    None => 0u8,
                    Some(v) => match &col.encoding {
                        ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } => {
                            encode_uint(&mut scratch.data, *signed, *varint, *bits, v)?;
                            1u8
                        }
                        ColumnEncoding::IntDict { .. } => encode_int_dict(
                            &mut scratch.data,
                            col,
                            self.dict_maps[idx].as_ref(),
                            (v) as i64,
                        )?,
                        _ => return Err(anyhow!("rowpack uint64 encoding mismatch")),
                    },
                },
                IngestedType::UInt16 => match access_or_null(access.get_uint32(idx)) {
                    None => 0u8,
                    Some(v) => match &col.encoding {
                        ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } => {
                            encode_uint(&mut scratch.data, *signed, *varint, *bits, v as u64)?;
                            1u8
                        }
                        ColumnEncoding::IntDict { .. } => encode_int_dict(
                            &mut scratch.data,
                            col,
                            self.dict_maps[idx].as_ref(),
                            (v as u64) as i64,
                        )?,
                        _ => return Err(anyhow!("rowpack uint16 encoding mismatch")),
                    },
                },
                IngestedType::UInt8 => match access_or_null(access.get_uint32(idx)) {
                    None => 0u8,
                    Some(v) => match &col.encoding {
                        ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } => {
                            encode_uint(&mut scratch.data, *signed, *varint, *bits, v as u64)?;
                            1u8
                        }
                        ColumnEncoding::IntDict { .. } => encode_int_dict(
                            &mut scratch.data,
                            col,
                            self.dict_maps[idx].as_ref(),
                            (v as u64) as i64,
                        )?,
                        _ => return Err(anyhow!("rowpack uint8 encoding mismatch")),
                    },
                },
                IngestedType::UInt32 => match access_or_null(access.get_uint32(idx)) {
                    None => 0u8,
                    Some(v) => match &col.encoding {
                        ColumnEncoding::Int {
                            signed,
                            varint,
                            bits,
                        } => {
                            encode_uint(&mut scratch.data, *signed, *varint, *bits, v as u64)?;
                            1u8
                        }
                        ColumnEncoding::IntDict { .. } => encode_int_dict(
                            &mut scratch.data,
                            col,
                            self.dict_maps[idx].as_ref(),
                            (v as u64) as i64,
                        )?,
                        _ => return Err(anyhow!("rowpack uint32 encoding mismatch")),
                    },
                },
                IngestedType::Float64 => match access_or_null(access.get_float64(idx)) {
                    None => 0u8,
                    Some(v) => {
                        match &col.encoding {
                            ColumnEncoding::Float { storage } => encode_float_with_tag(
                                &mut scratch.data,
                                col.logical_type,
                                *storage,
                                v,
                            )?,
                            ColumnEncoding::FloatDict { .. } => encode_float_dict(
                                &mut scratch.data,
                                col,
                                self.dict_maps[idx].as_ref(),
                                v,
                            )?,
                            _ => return Err(anyhow!("rowpack float64 encoding mismatch")),
                        }
                    }
                },
                IngestedType::Float32 => match access_or_null(access.get_float32(idx)) {
                    None => 0u8,
                    Some(v) => {
                        match &col.encoding {
                            ColumnEncoding::Float { storage } => encode_float_with_tag(
                                &mut scratch.data,
                                col.logical_type,
                                *storage,
                                v as f64,
                            )?,
                            ColumnEncoding::FloatDict { .. } => encode_float_dict(
                                &mut scratch.data,
                                col,
                                self.dict_maps[idx].as_ref(),
                                v as f64,
                            )?,
                            _ => return Err(anyhow!("rowpack float32 encoding mismatch")),
                        }
                    }
                },
                IngestedType::String => {
                    #[allow(clippy::manual_unwrap_or_default)]
                    match access.with_string(idx, |v| {
                        Ok(encode_string(
                            &mut scratch.data,
                            v,
                            col,
                            self.dict_maps[idx].as_ref(),
                            self.alphabet_maps[idx].as_ref(),
                            self.bpe_tries[idx].as_ref(),
                            Some(&mut scratch.bpe_tokens),
                            Some(&mut scratch.lz4_buf),
                        ))
                    }) {
                        Ok(Some(Ok(tag))) => tag,
                        Ok(Some(Err(err))) => return Err(err),
                        Ok(None) | Err(_) => 0u8,
                    }
                }
                IngestedType::Bytes =>
                {
                    #[allow(clippy::manual_unwrap_or_default)]
                    if let ColumnEncoding::Bytes { fixed_len } = &col.encoding {
                        match access.with_bytes(idx, |v| {
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
                        }) {
                            Ok(Some(tag)) => tag,
                            Ok(None) | Err(_) => 0u8,
                        }
                    } else {
                        return Err(anyhow!("rowpack bytes encoding mismatch"));
                    }
                }
                IngestedType::Uuid => match access_or_null(access.get_uuid(idx)) {
                    None => 0u8,
                    Some(v) => {
                        if !matches!(col.encoding, ColumnEncoding::Uuid) {
                            return Err(anyhow!("rowpack uuid encoding mismatch"));
                        }
                        scratch.data.extend_from_slice(&v);
                        1u8
                    }
                },
                IngestedType::Uuid256 => match access_or_null(access.get_uuid256(idx)) {
                    None => 0u8,
                    Some(v) => {
                        if !matches!(col.encoding, ColumnEncoding::Uuid256) {
                            return Err(anyhow!("rowpack uuid256 encoding mismatch"));
                        }
                        scratch.data.extend_from_slice(&v);
                        1u8
                    }
                },
                IngestedType::Ip => match access_or_null(access.get_ip(idx)) {
                    None => 0u8,
                    Some(v) => {
                        if !matches!(col.encoding, ColumnEncoding::Ip) {
                            return Err(anyhow!("rowpack ip encoding mismatch"));
                        }
                        scratch.data.extend_from_slice(&v);
                        1u8
                    }
                },
                IngestedType::Date => match access_or_null(access.get_date(idx)) {
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
                IngestedType::Timestamp => match access_or_null(access.get_timestamp(idx)) {
                    None => 0u8,
                    Some(secs) => {
                        write_varint(zigzag_encode(secs), &mut scratch.data);
                        1u8
                    }
                },
                IngestedType::Decimal128 => match access_or_null(access.get_decimal128(idx)) {
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
                IngestedType::Decimal256 => match access_or_null(access.get_decimal256(idx)) {
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
                IngestedType::Float32Array => match access.float_list_view(idx) {
                    Ok(None) | Err(_) => 0u8,
                    Ok(Some(view)) => {
                        if let Some((storage, fixed_len)) = float_array_encoding(&col.encoding) {
                            if !fixed_float_list_is_valid(&view, fixed_len) {
                                0u8
                            } else {
                                let f16_buf = &mut scratch.f16_buf;
                                encode_float_list_with_tag(
                                    &mut scratch.data,
                                    col.logical_type,
                                    storage,
                                    fixed_len,
                                    view,
                                    f16_buf,
                                )?
                            }
                        } else {
                            return Err(anyhow!("rowpack float32 array encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float16Array => match access.float_list_view(idx) {
                    Ok(None) | Err(_) => 0u8,
                    Ok(Some(view)) => {
                        if let Some((storage, fixed_len)) = float_array_encoding(&col.encoding) {
                            if !fixed_float_list_is_valid(&view, fixed_len) {
                                0u8
                            } else {
                                let f16_buf = &mut scratch.f16_buf;
                                encode_float_list_with_tag(
                                    &mut scratch.data,
                                    col.logical_type,
                                    storage,
                                    fixed_len,
                                    view,
                                    f16_buf,
                                )?
                            }
                        } else {
                            return Err(anyhow!("rowpack float16 array encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float64Array => match access.float_list_view(idx) {
                    Ok(None) | Err(_) => 0u8,
                    Ok(Some(view)) => {
                        if let Some((storage, fixed_len)) = float_array_encoding(&col.encoding) {
                            if !fixed_float_list_is_valid(&view, fixed_len) {
                                0u8
                            } else {
                                let f16_buf = &mut scratch.f16_buf;
                                encode_float_list_with_tag(
                                    &mut scratch.data,
                                    col.logical_type,
                                    storage,
                                    fixed_len,
                                    view,
                                    f16_buf,
                                )?
                            }
                        } else {
                            return Err(anyhow!("rowpack float64 array encoding mismatch"));
                        }
                    }
                },
                IngestedType::Int8Array
                | IngestedType::Int16Array
                | IngestedType::Int32Array
                | IngestedType::Int64Array => match access.int_list_view(idx) {
                    Ok(None) | Err(_) => 0u8,
                    Ok(Some(view)) => {
                        let fixed_len = int_array_fixed_len(&col.encoding);
                        if !fixed_int_list_is_valid(&view, fixed_len) {
                            0u8
                        } else {
                            let target_bits = self.int_array_target_bits[idx]
                                .ok_or_else(|| anyhow!("rowpack int-array bits unset"))?;
                            let row = access.row();
                            let col_label = col.name.as_str();
                            view.require_fits_in_signed(target_bits, col_label, row)?;
                            encode_int_list_view_with_tag(&mut scratch.data, view, fixed_len)?;
                            1u8
                        }
                    }
                },
                IngestedType::StringArray => match access.string_list_view(idx) {
                    Ok(None) | Err(_) => 0u8,
                    Ok(Some(view)) => {
                        if matches!(col.encoding, ColumnEncoding::StringArray { .. }) {
                            encode_string_list_view_with_tag(
                                &mut scratch.data,
                                &view,
                                col,
                                self.dict_maps[idx].as_ref(),
                                self.alphabet_maps[idx].as_ref(),
                                self.bpe_tries[idx].as_ref(),
                                Some(&mut scratch.bpe_tokens),
                                Some(&mut scratch.lz4_buf),
                            )?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack string array encoding mismatch"));
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
        self.encode_row_observed(
            access,
            tags,
            bools,
            data,
            f16_buf,
            key_bufs,
            None,
            &mut NoOpObserver,
        )
    }

    /// Generic version of `encode_row_from_access_into_buffers` that accepts a
    /// `ColumnValueObserver`. With `NoOpObserver`, LLVM monomorphizes away all
    /// observer calls — the generated code is identical to the non-generic path.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_row_observed<O: ColumnValueObserver>(
        &self,
        access: &mut RowAccess<'_>,
        tags: &mut Vec<u8>,
        bools: &mut Vec<u8>,
        data: &mut Vec<u8>,
        f16_buf: &mut Vec<f16>,
        key_bufs: &[Vec<u8>],
        // FK-free LP resolution overrides, indexed by column. `None` (the
        // common case) means every column encodes from `access` as usual.
        // When `Some(slice)`, a non-`None` entry at column `idx` supplies the
        // resolved node-id(s) directly (see [`ResolveOverride`]).
        resolve_overrides: Option<&[ResolveOverride]>,
        observer: &mut O,
    ) -> Result<RowPackRowLens> {
        debug_assert_eq!(key_bufs.len(), self.ctx.columns.len());
        debug_assert!(
            resolve_overrides
                .map(|o| o.len() == self.ctx.columns.len())
                .unwrap_or(true),
            "resolve_overrides length must match column count"
        );
        let tag_start = tags.len();
        let bool_start = bools.len();
        let data_start = data.len();
        let mut tag_writer = BitWriterRef::new(tags);
        let mut bool_writer = BitWriterRef::new(bools);

        for (idx, col) in self.ctx.columns.iter().enumerate() {
            let tag = match col.logical_type {
                IngestedType::Bool => {
                    match access_or_parse_failure(access.get_bool(idx), idx, observer) {
                        None => 0u8,
                        Some(v) => {
                            observer.observe_i64(idx, i64::from(v));
                            bool_writer.write_bits(if v { 1 } else { 0 }, 1);
                            1u8
                        }
                    }
                }
                IngestedType::Int64 => {
                    match access_or_parse_failure(access.get_int64(idx), idx, observer) {
                        None => 0u8,
                        Some(v) => {
                            observer.observe_i64(idx, v);
                            match &col.encoding {
                                ColumnEncoding::Int {
                                    signed,
                                    varint,
                                    bits,
                                } => {
                                    encode_int(data, *signed, *varint, *bits, v as i128)?;
                                    1u8
                                }
                                ColumnEncoding::IntDict { .. } => encode_int_dict(
                                    data,
                                    col,
                                    self.dict_maps[idx].as_ref(),
                                    (v as i128) as i64,
                                )?,
                                _ => return Err(anyhow!("rowpack int64 encoding mismatch")),
                            }
                        }
                    }
                }
                IngestedType::Int16 => {
                    match access_or_parse_failure(access.get_int32(idx), idx, observer) {
                        None => 0u8,
                        Some(v) => {
                            observer.observe_i64(idx, v as i64);
                            match &col.encoding {
                                ColumnEncoding::Int {
                                    signed,
                                    varint,
                                    bits,
                                } => {
                                    encode_int(data, *signed, *varint, *bits, v as i128)?;
                                    1u8
                                }
                                ColumnEncoding::IntDict { .. } => encode_int_dict(
                                    data,
                                    col,
                                    self.dict_maps[idx].as_ref(),
                                    (v as i128) as i64,
                                )?,
                                _ => return Err(anyhow!("rowpack int16 encoding mismatch")),
                            }
                        }
                    }
                }
                IngestedType::Int8 => {
                    match access_or_parse_failure(access.get_int32(idx), idx, observer) {
                        None => 0u8,
                        Some(v) => {
                            observer.observe_i64(idx, v as i64);
                            match &col.encoding {
                                ColumnEncoding::Int {
                                    signed,
                                    varint,
                                    bits,
                                } => {
                                    encode_int(data, *signed, *varint, *bits, v as i128)?;
                                    1u8
                                }
                                ColumnEncoding::IntDict { .. } => encode_int_dict(
                                    data,
                                    col,
                                    self.dict_maps[idx].as_ref(),
                                    (v as i128) as i64,
                                )?,
                                _ => return Err(anyhow!("rowpack int8 encoding mismatch")),
                            }
                        }
                    }
                }
                IngestedType::Int32 => {
                    match access_or_parse_failure(access.get_int32(idx), idx, observer) {
                        None => 0u8,
                        Some(v) => {
                            observer.observe_i64(idx, v as i64);
                            match &col.encoding {
                                ColumnEncoding::Int {
                                    signed,
                                    varint,
                                    bits,
                                } => {
                                    encode_int(data, *signed, *varint, *bits, v as i128)?;
                                    1u8
                                }
                                ColumnEncoding::IntDict { .. } => encode_int_dict(
                                    data,
                                    col,
                                    self.dict_maps[idx].as_ref(),
                                    (v as i128) as i64,
                                )?,
                                _ => return Err(anyhow!("rowpack int32 encoding mismatch")),
                            }
                        }
                    }
                }
                IngestedType::UInt64 => {
                    // FK-free LP resolution: if a scalar node-id override is
                    // present for this column, encode it directly (varint via
                    // encode_uint, mirroring the normal arm) instead of reading
                    // the source parquet, which has no such column.
                    let scalar_override = resolve_overrides
                        .and_then(|o| o.get(idx))
                        .and_then(|o| match o {
                            ResolveOverride::Scalar(v) => Some(*v),
                            _ => None,
                        });
                    let value = match scalar_override {
                        Some(v) => Some(v),
                        None => access_or_parse_failure(access.get_uint64(idx), idx, observer),
                    };
                    match value {
                        None => 0u8,
                        Some(v) => {
                            observer.observe_i64(idx, v as i64);
                            match &col.encoding {
                                ColumnEncoding::Int {
                                    signed,
                                    varint,
                                    bits,
                                } => {
                                    encode_uint(data, *signed, *varint, *bits, v)?;
                                    1u8
                                }
                                ColumnEncoding::IntDict { .. } => encode_int_dict(
                                    data,
                                    col,
                                    self.dict_maps[idx].as_ref(),
                                    (v) as i64,
                                )?,
                                _ => return Err(anyhow!("rowpack uint64 encoding mismatch")),
                            }
                        }
                    }
                }
                IngestedType::UInt16 => {
                    match access_or_parse_failure(access.get_uint32(idx), idx, observer) {
                        None => 0u8,
                        Some(v) => {
                            observer.observe_i64(idx, v as i64);
                            match &col.encoding {
                                ColumnEncoding::Int {
                                    signed,
                                    varint,
                                    bits,
                                } => {
                                    encode_uint(data, *signed, *varint, *bits, v as u64)?;
                                    1u8
                                }
                                ColumnEncoding::IntDict { .. } => encode_int_dict(
                                    data,
                                    col,
                                    self.dict_maps[idx].as_ref(),
                                    (v as u64) as i64,
                                )?,
                                _ => return Err(anyhow!("rowpack uint16 encoding mismatch")),
                            }
                        }
                    }
                }
                IngestedType::UInt8 => {
                    match access_or_parse_failure(access.get_uint32(idx), idx, observer) {
                        None => 0u8,
                        Some(v) => {
                            observer.observe_i64(idx, v as i64);
                            match &col.encoding {
                                ColumnEncoding::Int {
                                    signed,
                                    varint,
                                    bits,
                                } => {
                                    encode_uint(data, *signed, *varint, *bits, v as u64)?;
                                    1u8
                                }
                                ColumnEncoding::IntDict { .. } => encode_int_dict(
                                    data,
                                    col,
                                    self.dict_maps[idx].as_ref(),
                                    (v as u64) as i64,
                                )?,
                                _ => return Err(anyhow!("rowpack uint8 encoding mismatch")),
                            }
                        }
                    }
                }
                IngestedType::UInt32 => {
                    match access_or_parse_failure(access.get_uint32(idx), idx, observer) {
                        None => 0u8,
                        Some(v) => {
                            observer.observe_i64(idx, v as i64);
                            match &col.encoding {
                                ColumnEncoding::Int {
                                    signed,
                                    varint,
                                    bits,
                                } => {
                                    encode_uint(data, *signed, *varint, *bits, v as u64)?;
                                    1u8
                                }
                                ColumnEncoding::IntDict { .. } => encode_int_dict(
                                    data,
                                    col,
                                    self.dict_maps[idx].as_ref(),
                                    (v as u64) as i64,
                                )?,
                                _ => return Err(anyhow!("rowpack uint32 encoding mismatch")),
                            }
                        }
                    }
                }
                IngestedType::Float64 => {
                    match access_or_parse_failure(access.get_float64(idx), idx, observer) {
                        None => 0u8,
                        Some(v) => {
                            observer.observe_f64(idx, v);
                            match &col.encoding {
                                ColumnEncoding::Float { storage } => {
                                    encode_float_with_tag(data, col.logical_type, *storage, v)?
                                }
                                ColumnEncoding::FloatDict { .. } => {
                                    encode_float_dict(data, col, self.dict_maps[idx].as_ref(), v)?
                                }
                                _ => return Err(anyhow!("rowpack float64 encoding mismatch")),
                            }
                        }
                    }
                }
                IngestedType::Float32 => {
                    match access_or_parse_failure(access.get_float32(idx), idx, observer) {
                        None => 0u8,
                        Some(v) => {
                            observer.observe_f64(idx, v as f64);
                            match &col.encoding {
                                ColumnEncoding::Float { storage } => encode_float_with_tag(
                                    data,
                                    col.logical_type,
                                    *storage,
                                    v as f64,
                                )?,
                                ColumnEncoding::FloatDict { .. } => encode_float_dict(
                                    data,
                                    col,
                                    self.dict_maps[idx].as_ref(),
                                    v as f64,
                                )?,
                                _ => return Err(anyhow!("rowpack float32 encoding mismatch")),
                            }
                        }
                    }
                }
                IngestedType::String => {
                    #[allow(clippy::manual_unwrap_or_default)]
                    match access.with_string(idx, |v| {
                        observer.observe_string(idx, v);
                        Ok(encode_string(
                            data,
                            v,
                            col,
                            self.dict_maps[idx].as_ref(),
                            self.alphabet_maps[idx].as_ref(),
                            self.bpe_tries[idx].as_ref(),
                            None,
                            None,
                        ))
                    }) {
                        Ok(Some(Ok(tag))) => tag,
                        Ok(Some(Err(err))) => return Err(err),
                        Ok(None) => 0u8,
                        Err(_) => {
                            observer.observe_parse_failure(idx);
                            0u8
                        }
                    }
                }
                IngestedType::Bytes =>
                {
                    #[allow(clippy::manual_unwrap_or_default)]
                    if let ColumnEncoding::Bytes { fixed_len } = &col.encoding {
                        match access.with_bytes(idx, |v| {
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
                        }) {
                            Ok(Some(tag)) => tag,
                            Ok(None) => 0u8,
                            Err(_) => {
                                observer.observe_parse_failure(idx);
                                0u8
                            }
                        }
                    } else {
                        return Err(anyhow!("rowpack bytes encoding mismatch"));
                    }
                }
                IngestedType::Uuid => {
                    let buf = &key_bufs[idx];
                    if buf.len() == 16 {
                        data.extend_from_slice(buf);
                        1u8
                    } else {
                        match access_or_parse_failure(access.get_uuid(idx), idx, observer) {
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
                        match access_or_parse_failure(access.get_uuid256(idx), idx, observer) {
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
                        match access_or_parse_failure(access.get_ip(idx), idx, observer) {
                            None => 0u8,
                            Some(v) => {
                                data.extend_from_slice(&v);
                                1u8
                            }
                        }
                    }
                }
                IngestedType::Date => {
                    match access_or_parse_failure(access.get_date(idx), idx, observer) {
                        None => 0u8,
                        Some(v) => {
                            observer.observe_i64(idx, v as i64);
                            if let ColumnEncoding::Date { varint, bits } = &col.encoding {
                                encode_int(data, true, *varint, *bits, v as i128)?;
                                1u8
                            } else {
                                return Err(anyhow!("rowpack date encoding mismatch"));
                            }
                        }
                    }
                }
                IngestedType::Timestamp => match access.get_timestamp(idx) {
                    Ok(None) => 0u8,
                    Ok(Some(secs)) => {
                        observer.observe_timestamp(idx, secs);
                        write_varint(zigzag_encode(secs), data);
                        1u8
                    }
                    Err(_) => {
                        observer.observe_parse_failure(idx);
                        0u8
                    }
                },
                IngestedType::Decimal128 => {
                    match access_or_parse_failure(access.get_decimal128(idx), idx, observer) {
                        None => 0u8,
                        Some(v) => {
                            if let ColumnEncoding::Decimal {
                                bytes,
                                scale,
                                storage,
                                ..
                            } = &col.encoding
                            {
                                observer.observe_f64(idx, v as f64 / 10_f64.powi(*scale));
                                if *bytes == 0 || *bytes > 16 {
                                    return Err(anyhow!("rowpack decimal size mismatch"));
                                }
                                encode_decimal_i128_with_tag(data, v, *scale, storage, *bytes)?
                            } else {
                                return Err(anyhow!("rowpack decimal128 encoding mismatch"));
                            }
                        }
                    }
                }
                IngestedType::Decimal256 => {
                    match access_or_parse_failure(access.get_decimal256(idx), idx, observer) {
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
                    }
                }
                IngestedType::Float32Array => match access.float_list_view(idx) {
                    Ok(None) => 0u8,
                    Err(_) => {
                        observer.observe_parse_failure(idx);
                        0u8
                    }
                    Ok(Some(view)) => {
                        if let Some((storage, fixed_len)) = float_array_encoding(&col.encoding) {
                            if !fixed_float_list_is_valid(&view, fixed_len) {
                                observer.observe_parse_failure(idx);
                                0u8
                            } else {
                                observer.observe_float_list(idx, &view);
                                encode_float_list_with_tag(
                                    data,
                                    col.logical_type,
                                    storage,
                                    fixed_len,
                                    view,
                                    f16_buf,
                                )?
                            }
                        } else {
                            return Err(anyhow!("rowpack float32 array encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float16Array => match access.float_list_view(idx) {
                    Ok(None) => 0u8,
                    Err(_) => {
                        observer.observe_parse_failure(idx);
                        0u8
                    }
                    Ok(Some(view)) => {
                        if let Some((storage, fixed_len)) = float_array_encoding(&col.encoding) {
                            if !fixed_float_list_is_valid(&view, fixed_len) {
                                observer.observe_parse_failure(idx);
                                0u8
                            } else {
                                observer.observe_float_list(idx, &view);
                                encode_float_list_with_tag(
                                    data,
                                    col.logical_type,
                                    storage,
                                    fixed_len,
                                    view,
                                    f16_buf,
                                )?
                            }
                        } else {
                            return Err(anyhow!("rowpack float16 array encoding mismatch"));
                        }
                    }
                },
                IngestedType::Float64Array => match access.float_list_view(idx) {
                    Ok(None) => 0u8,
                    Err(_) => {
                        observer.observe_parse_failure(idx);
                        0u8
                    }
                    Ok(Some(view)) => {
                        if let Some((storage, fixed_len)) = float_array_encoding(&col.encoding) {
                            if !fixed_float_list_is_valid(&view, fixed_len) {
                                observer.observe_parse_failure(idx);
                                0u8
                            } else {
                                observer.observe_float_list(idx, &view);
                                encode_float_list_with_tag(
                                    data,
                                    col.logical_type,
                                    storage,
                                    fixed_len,
                                    view,
                                    f16_buf,
                                )?
                            }
                        } else {
                            return Err(anyhow!("rowpack float64 array encoding mismatch"));
                        }
                    }
                },
                IngestedType::Int8Array
                | IngestedType::Int16Array
                | IngestedType::Int32Array
                | IngestedType::Int64Array => {
                    let fixed_len = int_array_fixed_len(&col.encoding);
                    // FK-free LP resolution: if a list node-id override is
                    // present, wrap the resolved ids as a `Parsed` IntListView
                    // and run the existing list encoder. The column's encoding
                    // is bits:64 (Int64Array), so require_fits_in_signed(64) is
                    // a no-op and node-ids are never rejected.
                    let list_override = resolve_overrides
                        .and_then(|o| o.get(idx))
                        .and_then(|o| match o {
                            ResolveOverride::List(ids) => Some(ids),
                            _ => None,
                        });
                    if let Some(ids) = list_override {
                        let view = IntListView {
                            start: 0,
                            end: ids.len(),
                            values: IntListValues::Parsed(ids.clone()),
                        };
                        observer.observe_int64_list(idx, &view);
                        let target_bits = self.int_array_target_bits[idx]
                            .ok_or_else(|| anyhow!("rowpack int-array bits unset"))?;
                        view.require_fits_in_signed(target_bits, col.name.as_str(), access.row())?;
                        encode_int_list_view_with_tag(data, view, fixed_len)?;
                        1u8
                    } else {
                        match access.int_list_view(idx) {
                            Ok(None) => 0u8,
                            Err(_) => {
                                observer.observe_parse_failure(idx);
                                0u8
                            }
                            Ok(Some(view)) => {
                                if !fixed_int_list_is_valid(&view, fixed_len) {
                                    observer.observe_parse_failure(idx);
                                    0u8
                                } else {
                                    observer.observe_int64_list(idx, &view);
                                    let target_bits = self.int_array_target_bits[idx]
                                        .ok_or_else(|| anyhow!("rowpack int-array bits unset"))?;
                                    let row = access.row();
                                    let col_label = col.name.as_str();
                                    view.require_fits_in_signed(target_bits, col_label, row)?;
                                    encode_int_list_view_with_tag(data, view, fixed_len)?;
                                    1u8
                                }
                            }
                        }
                    }
                }
                IngestedType::StringArray => match access.string_list_view(idx) {
                    Ok(None) => 0u8,
                    Err(_) => {
                        observer.observe_parse_failure(idx);
                        0u8
                    }
                    Ok(Some(view)) => {
                        observer.observe_string_list(idx, &view);
                        if matches!(col.encoding, ColumnEncoding::StringArray { .. }) {
                            encode_string_list_view_with_tag(
                                data,
                                &view,
                                col,
                                self.dict_maps[idx].as_ref(),
                                self.alphabet_maps[idx].as_ref(),
                                self.bpe_tries[idx].as_ref(),
                                None,
                                None,
                            )?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack string array encoding mismatch"));
                        }
                    }
                },
            };
            if tag == 0 {
                observer.observe_null(idx);
            }
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
    encode_string_value(out, value, encoding, _dict, _alphabet, _bpe, _bpe_tokens, _lz4_buf)
}

/// Encodes one string against an encoding directly. List columns hold their
/// element encoding inline, and building a `ColumnContext` for each element
/// would clone that encoding -- which for a dictionary is one allocation per
/// entry, per row.
#[allow(clippy::too_many_arguments)]
fn encode_string_value(
    out: &mut Vec<u8>,
    value: &str,
    encoding: &StringEncoding,
    _dict: Option<&FastHashMap<Vec<u8>, u32>>,
    _alphabet: Option<&AlphabetMap>,
    _bpe: Option<&BpeTrie>,
    _bpe_tokens: Option<&mut Vec<u32>>,
    _lz4_buf: Option<&mut Vec<u8>>,
) -> Result<u8> {
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

fn encode_int_dict(
    out: &mut Vec<u8>,
    col: &ColumnContext,
    dict: Option<&FastHashMap<Vec<u8>, u32>>,
    v: i64,
) -> Result<u8> {
    let ColumnEncoding::IntDict {
        signed,
        varint,
        bits,
        ..
    } = &col.encoding
    else {
        return Err(anyhow!("rowpack int dict encoding mismatch"));
    };
    let map = dict.ok_or_else(|| anyhow!("rowpack missing int dict"))?;
    match map.get(v.to_le_bytes().as_slice()) {
        Some(idx) => {
            write_varint(*idx as u64, out);
            Ok(1)
        }
        None => {
            encode_int(out, *signed, *varint, *bits, v as i128)?;
            Ok(2)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_value_with_tag<'a>(
    out: &mut Vec<u8>,
    col: &ColumnContext,
    value: &ValueRef<'a>,
    dict: Option<&FastHashMap<Vec<u8>, u32>>,
    alphabet: Option<&AlphabetMap>,
    bpe: Option<&BpeTrie>,
    bpe_tokens: Option<&mut Vec<u32>>,
    lz4_buf: Option<&mut Vec<u8>>,
) -> Result<u8> {
    match (&col.encoding, value) {
        // `ValueRef` only carries 32/64-bit ints; narrower widths arrive widened.
        (
            ColumnEncoding::IntDict { .. },
            ValueRef::Int64(_) | ValueRef::Int32(_) | ValueRef::UInt64(_) | ValueRef::UInt32(_),
        ) => {
            let v = match value {
                ValueRef::Int64(v) => *v,
                ValueRef::Int32(v) => *v as i64,
                ValueRef::UInt64(v) => *v as i64,
                ValueRef::UInt32(v) => *v as i64,
                _ => unreachable!("guarded by the pattern above"),
            };
            encode_int_dict(out, col, dict, v)
        }
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
        (ColumnEncoding::FloatDict { .. }, ValueRef::Float64(v)) => {
            encode_float_dict(out, col, dict, *v)
        }
        (ColumnEncoding::FloatDict { .. }, ValueRef::Float32(v)) => {
            encode_float_dict(out, col, dict, *v as f64)
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
        (ColumnEncoding::Timestamp, ValueRef::Timestamp(secs)) => {
            write_varint(zigzag_encode(*secs), out);
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
            let tag =
                encode_float_slice_f32_with_tag(out, col.logical_type, *storage, None, values)?;
            Ok(tag)
        }
        (ColumnEncoding::FixedFloatArray { storage, len }, ValueRef::Float32Array(values)) => {
            let tag = encode_float_slice_f32_with_tag(
                out,
                col.logical_type,
                *storage,
                Some(*len),
                values,
            )?;
            Ok(tag)
        }
        (ColumnEncoding::FloatArray { storage }, ValueRef::Float64Array(values)) => {
            let tag =
                encode_float_slice_f64_with_tag(out, col.logical_type, *storage, None, values)?;
            Ok(tag)
        }
        (ColumnEncoding::FixedFloatArray { storage, len }, ValueRef::Float64Array(values)) => {
            let tag = encode_float_slice_f64_with_tag(
                out,
                col.logical_type,
                *storage,
                Some(*len),
                values,
            )?;
            Ok(tag)
        }
        (ColumnEncoding::Int8Array, ValueRef::Int8Array(values)) => {
            encode_int_slice_with_tag(out, values.iter().map(|&v| v as i64), None)?;
            Ok(1)
        }
        (ColumnEncoding::FixedIntArray { len }, ValueRef::Int8Array(values))
            if col.logical_type == IngestedType::Int8Array =>
        {
            encode_int_slice_with_tag(out, values.iter().map(|&v| v as i64), Some(*len))?;
            Ok(1)
        }
        (ColumnEncoding::Int16Array, ValueRef::Int16Array(values)) => {
            encode_int_slice_with_tag(out, values.iter().map(|&v| v as i64), None)?;
            Ok(1)
        }
        (ColumnEncoding::FixedIntArray { len }, ValueRef::Int16Array(values))
            if col.logical_type == IngestedType::Int16Array =>
        {
            encode_int_slice_with_tag(out, values.iter().map(|&v| v as i64), Some(*len))?;
            Ok(1)
        }
        (ColumnEncoding::Int32Array, ValueRef::Int32Array(values)) => {
            encode_int_slice_with_tag(out, values.iter().map(|&v| v as i64), None)?;
            Ok(1)
        }
        (ColumnEncoding::FixedIntArray { len }, ValueRef::Int32Array(values))
            if col.logical_type == IngestedType::Int32Array =>
        {
            encode_int_slice_with_tag(out, values.iter().map(|&v| v as i64), Some(*len))?;
            Ok(1)
        }
        (ColumnEncoding::Int64Array, ValueRef::Int64Array(values)) => {
            encode_int64_list_with_tag(out, values, None)?;
            Ok(1)
        }
        (ColumnEncoding::FixedIntArray { len }, ValueRef::Int64Array(values))
            if col.logical_type == IngestedType::Int64Array =>
        {
            encode_int64_list_with_tag(out, values, Some(*len))?;
            Ok(1)
        }
        (ColumnEncoding::StringArray { .. }, ValueRef::StringArray(tokens)) => {
            encode_string_list_with_tag(
                out, tokens, col, dict, alphabet, bpe, bpe_tokens, lz4_buf,
            )?;
            Ok(1)
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

fn encode_float_dict(
    out: &mut Vec<u8>,
    col: &ColumnContext,
    dict: Option<&FastHashMap<Vec<u8>, u32>>,
    v: f64,
) -> Result<u8> {
    let ColumnEncoding::FloatDict { storage, .. } = &col.encoding else {
        return Err(anyhow!("rowpack float dict encoding mismatch"));
    };
    let map = dict.ok_or_else(|| anyhow!("rowpack missing float dict"))?;
    match map.get(v.to_le_bytes().as_slice()) {
        Some(idx) => {
            write_varint(*idx as u64, out);
            Ok(1)
        }
        None => {
            encode_float(out, *storage, v)?;
            Ok(2)
        }
    }
}

fn decode_float_dict(
    values: &[f64],
    storage: FloatStorage,
    tag: u8,
    data: &mut &[u8],
) -> Result<f64> {
    if tag == 1 {
        let idx = read_varint(data)? as usize;
        return values
            .get(idx)
            .copied()
            .ok_or_else(|| anyhow!("rowpack float dict index {idx} out of range"));
    }
    decode_float(data, storage)
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

#[inline]
fn float_array_encoding(encoding: &ColumnEncoding) -> Option<(FloatStorage, Option<usize>)> {
    match encoding {
        ColumnEncoding::FloatArray { storage } => Some((*storage, None)),
        ColumnEncoding::FixedFloatArray { storage, len } => Some((*storage, Some(*len))),
        _ => None,
    }
}

#[inline]
fn fixed_float_list_is_valid(view: &FloatListView, fixed_len: Option<usize>) -> bool {
    fixed_len.is_none_or(|len| view.len() == len && !view.has_nulls())
}

#[inline]
fn int_array_fixed_len(encoding: &ColumnEncoding) -> Option<usize> {
    match encoding {
        ColumnEncoding::FixedIntArray { len } => Some(*len),
        _ => None,
    }
}

#[inline]
fn fixed_int_list_is_valid(view: &IntListView, fixed_len: Option<usize>) -> bool {
    fixed_len.is_none_or(|len| view.len() == len && !view.has_nulls())
}

#[inline]
fn array_len_prefix_bytes(fixed_len: Option<usize>, len: usize) -> usize {
    if fixed_len.is_some() {
        0
    } else {
        varint_bytes_u64(len as u64)
    }
}

fn encode_float_list_with_tag(
    out: &mut Vec<u8>,
    logical: IngestedType,
    storage: FloatStorage,
    fixed_len: Option<usize>,
    view: FloatListView,
    f16_buf: &mut Vec<f16>,
) -> Result<u8> {
    let _ = logical;
    let len = view.end.saturating_sub(view.start);
    if !fixed_float_list_is_valid(&view, fixed_len) {
        return Err(anyhow!("rowpack fixed float array length mismatch"));
    }
    let bytes_per = match storage {
        FloatStorage::F16 => 2,
        FloatStorage::F32 => 4,
        FloatStorage::F64 => 8,
    };
    out.reserve(array_len_prefix_bytes(fixed_len, len) + len * bytes_per);
    if fixed_len.is_none() {
        write_varint(len as u64, out);
    }
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
        FloatListValues::Parsed(ref parsed) => {
            // Parsed from string — clean Vec<f32>, no nulls.
            let slice = &parsed[view.start..view.end];
            match storage {
                FloatStorage::F16 => {
                    f16_buf.resize(slice.len(), f16::from_bits(0));
                    f16_buf.as_mut_slice().convert_from_f32_slice(slice);
                    extend_f16_slice_le(out, f16_buf);
                }
                FloatStorage::F32 => {
                    extend_f32_slice_le(out, slice);
                }
                FloatStorage::F64 => {
                    for &v in slice {
                        out.extend_from_slice(&(v as f64).to_le_bytes());
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
    fixed_len: Option<usize>,
    values: &[f32],
) -> Result<u8> {
    let _ = logical;
    if fixed_len.is_some_and(|len| values.len() != len) {
        return Err(anyhow!("rowpack fixed float array length mismatch"));
    }
    let bytes_per = match storage {
        FloatStorage::F16 => 2,
        FloatStorage::F32 => 4,
        FloatStorage::F64 => 8,
    };
    out.reserve(array_len_prefix_bytes(fixed_len, values.len()) + values.len() * bytes_per);
    if fixed_len.is_none() {
        write_varint(values.len() as u64, out);
    }
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
    fixed_len: Option<usize>,
    values: &[f64],
) -> Result<u8> {
    let _ = logical;
    if fixed_len.is_some_and(|len| values.len() != len) {
        return Err(anyhow!("rowpack fixed float array length mismatch"));
    }
    let bytes_per = match storage {
        FloatStorage::F16 => 2,
        FloatStorage::F32 => 4,
        FloatStorage::F64 => 8,
    };
    out.reserve(array_len_prefix_bytes(fixed_len, values.len()) + values.len() * bytes_per);
    if fixed_len.is_none() {
        write_varint(values.len() as u64, out);
    }
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

/// Encode a list of i64 values as: varint(N) + [zigzag_varint(v) × N].
/// Returns tag 1.
fn encode_int64_list_with_tag(
    out: &mut Vec<u8>,
    values: &[i64],
    fixed_len: Option<usize>,
) -> Result<()> {
    if fixed_len.is_some_and(|len| values.len() != len) {
        return Err(anyhow!("rowpack fixed int array length mismatch"));
    }
    out.reserve(array_len_prefix_bytes(fixed_len, values.len()) + values.len() * 2);
    if fixed_len.is_none() {
        write_varint(values.len() as u64, out);
    }
    for &v in values {
        write_varint(zigzag_encode(v), out);
    }
    Ok(())
}

/// Generic variant that takes an iterator yielding `i64`-widened values.
/// Used by the narrower `Int{8,16,32}Array` ValueRef encoders, which widen
/// their element type on the fly. The output bytes are identical to
/// `encode_int64_list_with_tag` for the same numeric values — the width tag
/// lives in `ColumnEncoding`, not in the wire format.
fn encode_int_slice_with_tag<I>(
    out: &mut Vec<u8>,
    values: I,
    fixed_len: Option<usize>,
) -> Result<()>
where
    I: ExactSizeIterator<Item = i64>,
{
    let len = values.len();
    if fixed_len.is_some_and(|expected| len != expected) {
        return Err(anyhow!("rowpack fixed int array length mismatch"));
    }
    out.reserve(array_len_prefix_bytes(fixed_len, len) + len * 2);
    if fixed_len.is_none() {
        write_varint(len as u64, out);
    }
    for v in values {
        write_varint(zigzag_encode(v), out);
    }
    Ok(())
}

/// Map an int-array `ColumnEncoding` (paired with the column's `IngestedType`)
/// to the target signed bit width, used by the encoder's overflow check.
/// Returns an error if the pair is internally inconsistent (a bug, not user
/// data, since infer_column_encoding builds these pairs in sync).
fn int_array_encoding_bits(encoding: &ColumnEncoding, logical: IngestedType) -> Result<u8> {
    let bits = match encoding {
        ColumnEncoding::Int8Array => 8,
        ColumnEncoding::Int16Array => 16,
        ColumnEncoding::Int32Array => 32,
        ColumnEncoding::Int64Array => 64,
        ColumnEncoding::FixedIntArray { .. } => match logical {
            IngestedType::Int8Array => 8,
            IngestedType::Int16Array => 16,
            IngestedType::Int32Array => 32,
            IngestedType::Int64Array => 64,
            _ => {
                return Err(anyhow!(
                    "rowpack int-array encoding mismatch: logical_type={:?} encoding={:?}",
                    logical,
                    encoding,
                ));
            }
        },
        other => {
            return Err(anyhow!(
                "rowpack int-array encoding mismatch: logical_type={:?} encoding={:?}",
                logical,
                other,
            ));
        }
    };
    let expected = match logical {
        IngestedType::Int8Array => 8,
        IngestedType::Int16Array => 16,
        IngestedType::Int32Array => 32,
        IngestedType::Int64Array => 64,
        other => {
            return Err(anyhow!(
                "rowpack int-array encoding mismatch: non-int-array logical type {:?}",
                other,
            ));
        }
    };
    if bits != expected {
        return Err(anyhow!(
            "rowpack int-array encoding mismatch: logical_type={:?} ({}b) vs encoding={:?} ({}b)",
            logical,
            expected,
            encoding,
            bits,
        ));
    }
    Ok(bits)
}

fn encode_int_list_view_with_tag(
    out: &mut Vec<u8>,
    view: IntListView,
    fixed_len: Option<usize>,
) -> Result<()> {
    if !fixed_int_list_is_valid(&view, fixed_len) {
        return Err(anyhow!("rowpack fixed int array length mismatch"));
    }
    // The wire format (varint(N) + zigzag-varint per non-null element) is the
    // same for every signed width because each element is widened to i64
    // before zigzag-encoding. Width is carried by the column's
    // ColumnEncoding tag, not in the bytes.
    fn encode_native<E, F>(
        out: &mut Vec<u8>,
        arr: &E,
        start: usize,
        end: usize,
        get: F,
        fixed_len: Option<usize>,
    ) -> Result<()>
    where
        E: arrow_array::Array,
        F: Fn(&E, usize) -> i64,
    {
        let total = end.saturating_sub(start);
        if arr.null_count() == 0 {
            out.reserve(array_len_prefix_bytes(fixed_len, total) + total * 2);
            if fixed_len.is_none() {
                write_varint(total as u64, out);
            }
            for i in start..end {
                write_varint(zigzag_encode(get(arr, i)), out);
            }
            return Ok(());
        }
        // Null elements are dropped: the persisted list length reflects
        // only non-null positions. A legitimate zero value is preserved;
        // only the validity bit filters.
        let mut non_null_count = 0usize;
        for i in start..end {
            if !arr.is_null(i) {
                non_null_count += 1;
            }
        }
        out.reserve(array_len_prefix_bytes(fixed_len, non_null_count) + non_null_count * 2);
        if fixed_len.is_none() {
            write_varint(non_null_count as u64, out);
        }
        for i in start..end {
            if !arr.is_null(i) {
                write_varint(zigzag_encode(get(arr, i)), out);
            }
        }
        Ok(())
    }

    match &view.values {
        IntListValues::I8(values) => {
            let arr = values
                .as_any()
                .downcast_ref::<arrow_array::Int8Array>()
                .ok_or_else(|| anyhow!("int8 list element cast"))?;
            encode_native(
                out,
                arr,
                view.start,
                view.end,
                |a, i| a.value(i) as i64,
                fixed_len,
            )
        }
        IntListValues::I16(values) => {
            let arr = values
                .as_any()
                .downcast_ref::<arrow_array::Int16Array>()
                .ok_or_else(|| anyhow!("int16 list element cast"))?;
            encode_native(
                out,
                arr,
                view.start,
                view.end,
                |a, i| a.value(i) as i64,
                fixed_len,
            )
        }
        IntListValues::I32(values) => {
            let arr = values
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .ok_or_else(|| anyhow!("int32 list element cast"))?;
            encode_native(
                out,
                arr,
                view.start,
                view.end,
                |a, i| a.value(i) as i64,
                fixed_len,
            )
        }
        IntListValues::I64(values) => {
            let arr = values
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .ok_or_else(|| anyhow!("int64 list element cast"))?;
            // Null-free path: borrow the values buffer as `&[i64]` and run the
            // tight unchecked-iteration encoder. This avoids `arr.value(i)`'s
            // per-element bounds-check assert that `encode_native` incurs.
            // The list-element type for `List<Int64>` is the dominant input
            // shape, so this fast path matters.
            if arr.null_count() == 0 {
                encode_int64_list_with_tag(out, &arr.values()[view.start..view.end], fixed_len)
            } else {
                encode_native(out, arr, view.start, view.end, |a, i| a.value(i), fixed_len)
            }
        }
        IntListValues::Parsed(values) => {
            encode_int64_list_with_tag(out, &values[view.start..view.end], fixed_len)
        }
    }
}

/// Encode a list of string tokens using the column's StringEncoding.
///
/// Wire format: `varint(N)` + for each token: `u8(per_token_tag)` + encoded bytes.
///   - per_token_tag=1: dict hit (varint index) or raw encoding succeeded
///   - per_token_tag=2: dict miss / fallback (varint(len) + raw bytes)
#[allow(clippy::too_many_arguments)]
fn encode_string_list_with_tag(
    out: &mut Vec<u8>,
    tokens: &[&str],
    col: &ColumnContext,
    dict: Option<&FastHashMap<Vec<u8>, u32>>,
    alphabet: Option<&AlphabetMap>,
    bpe: Option<&BpeTrie>,
    mut bpe_tokens: Option<&mut Vec<u32>>,
    mut lz4_buf: Option<&mut Vec<u8>>,
) -> Result<()> {
    let ColumnEncoding::StringArray { encoding } = &col.encoding else {
        return Err(anyhow!("rowpack string array encoding mismatch"));
    };
    write_varint(tokens.len() as u64, out);
    // Temporary buffer to hold one token's bytes before we know the tag.
    let mut token_buf = Vec::new();
    for &token in tokens {
        token_buf.clear();
        // Reborrow mutable references for each iteration so they aren't consumed.
        let bpe_tok_ref: Option<&mut Vec<u32>> = bpe_tokens.as_deref_mut();
        let lz4_ref: Option<&mut Vec<u8>> = lz4_buf.as_deref_mut();
        let token_tag = encode_string_value(
            &mut token_buf,
            token,
            encoding,
            dict,
            alphabet,
            bpe,
            bpe_tok_ref,
            lz4_ref,
        )?;
        out.push(token_tag);
        out.extend_from_slice(&token_buf);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_string_list_view_with_tag(
    out: &mut Vec<u8>,
    view: &StringListView,
    col: &ColumnContext,
    dict: Option<&FastHashMap<Vec<u8>, u32>>,
    alphabet: Option<&AlphabetMap>,
    bpe: Option<&BpeTrie>,
    mut bpe_tokens: Option<&mut Vec<u32>>,
    mut lz4_buf: Option<&mut Vec<u8>>,
) -> Result<()> {
    let ColumnEncoding::StringArray { encoding } = &col.encoding else {
        return Err(anyhow!("rowpack string array encoding mismatch"));
    };

    // Compute the non-null element count up-front. Null elements are dropped
    // from the persisted list; a legitimate empty string is preserved because
    // only the validity bit filters. For the Parsed variant (which carries no
    // per-element null information), every entry counts.
    let non_null_count: usize = match &view.values {
        StringListValues::Utf8(values) => {
            let values_arr = values
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| anyhow!("string list element cast"))?;
            (view.start..view.end)
                .filter(|&i| !values_arr.is_null(i))
                .count()
        }
        StringListValues::LargeUtf8(values) => {
            let values_arr = values
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .ok_or_else(|| anyhow!("string list element cast"))?;
            (view.start..view.end)
                .filter(|&i| !values_arr.is_null(i))
                .count()
        }
        StringListValues::Parsed(values) => values[view.start..view.end].len(),
    };
    write_varint(non_null_count as u64, out);

    let mut token_buf = Vec::new();
    let mut encode_token = |token: &str| -> Result<()> {
        token_buf.clear();
        let bpe_tok_ref: Option<&mut Vec<u32>> = bpe_tokens.as_deref_mut();
        let lz4_ref: Option<&mut Vec<u8>> = lz4_buf.as_deref_mut();
        let token_tag = encode_string_value(
            &mut token_buf,
            token,
            encoding,
            dict,
            alphabet,
            bpe,
            bpe_tok_ref,
            lz4_ref,
        )?;
        out.push(token_tag);
        out.extend_from_slice(&token_buf);
        Ok(())
    };

    match &view.values {
        StringListValues::Utf8(values) => {
            let values_arr = values
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| anyhow!("string list element cast"))?;
            for i in view.start..view.end {
                if !values_arr.is_null(i) {
                    encode_token(values_arr.value(i))?;
                }
            }
        }
        StringListValues::LargeUtf8(values) => {
            let values_arr = values
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .ok_or_else(|| anyhow!("string list element cast"))?;
            for i in view.start..view.end {
                if !values_arr.is_null(i) {
                    encode_token(values_arr.value(i))?;
                }
            }
        }
        StringListValues::Parsed(values) => {
            for token in &values[view.start..view.end] {
                encode_token(token)?;
            }
        }
    }
    Ok(())
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

/// Payload bytes `samples` would occupy under `encoding`, counting the same
/// per-value framing and fallbacks the encoders write, so candidates are
/// directly comparable. Excludes the per-column context, which the eligibility
/// rules already bound.
fn estimate_string_bytes(samples: &[Vec<u8>], encoding: &StringEncoding) -> u64 {
    match encoding {
        StringEncoding::Raw => estimate_raw_bytes(samples),
        // Alphabet is only ever a candidate when every observed value is in the
        // alphabet, so there is no unpacked fallback to price here.
        StringEncoding::Alphabet { alphabet } => estimate_alphabet_bytes(samples, alphabet.len()),
        StringEncoding::Dict { values } => estimate_dict_bytes(samples, values),
        StringEncoding::Lz4 => estimate_lz4_bytes(samples),
        StringEncoding::Bpe { dict } => estimate_bpe_bytes(samples, dict),
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

fn estimate_dict_bytes(samples: &[Vec<u8>], values: &[Vec<u8>]) -> u64 {
    let index: FastHashMap<&[u8], u32> = values
        .iter()
        .enumerate()
        .map(|(idx, value)| (value.as_slice(), idx as u32))
        .collect();
    let mut total = 0u64;
    for sample in samples {
        total += match index.get(sample.as_slice()) {
            Some(idx) => varint_bytes_u64(*idx as u64) as u64,
            None => varint_bytes_u64(sample.len() as u64) as u64 + sample.len() as u64,
        };
    }
    total
}

fn estimate_lz4_bytes(samples: &[Vec<u8>]) -> u64 {
    let mut buf = Vec::new();
    let mut total = 0u64;
    for sample in samples {
        let raw_total = varint_bytes_u64(sample.len() as u64) + sample.len();
        buf.resize(lz4_flex::block::get_maximum_output_size(sample.len()), 0);
        let compressed_total = match lz4_flex::block::compress_into(sample, &mut buf) {
            Ok(written) => {
                varint_bytes_u64(sample.len() as u64)
                    + varint_bytes_u64(written as u64)
                    + written
            }
            Err(_) => raw_total,
        };
        total += compressed_total.min(raw_total) as u64;
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
    clock: TimeUnit,
}

pub struct RowPackDecodeScratch {
    pub string: String,
    pub bytes: Vec<u8>,
    pub float32: Vec<f32>,
    pub float64: Vec<f64>,
    pub int8: Vec<i8>,
    pub int16: Vec<i16>,
    pub int32: Vec<i32>,
    pub int64: Vec<i64>,
    /// Decoded string tokens for StringArray columns.
    /// Each token is stored as a `String` and we return `&str` into them.
    pub strings: Vec<String>,
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
            int8: Vec::with_capacity(64),
            int16: Vec::with_capacity(64),
            int32: Vec::with_capacity(64),
            int64: Vec::with_capacity(64),
            strings: Vec::new(),
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
                }
                | ColumnEncoding::StringArray {
                    encoding: StringEncoding::Dict { values },
                } => dict_values.push(Some(values.clone())),
                _ => dict_values.push(None),
            }
        }
        Ok(Self {
            ctx,
            dict_values,
            clock: TimeUnit::Seconds,
        })
    }

    pub fn with_clock(mut self, clock: TimeUnit) -> Self {
        self.clock = clock;
        self
    }

    pub fn clock(&self) -> TimeUnit {
        self.clock
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
                self.clock,
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
                decode_value_json(
                    col,
                    tag,
                    &mut data,
                    self.dict_values[idx].as_ref(),
                    self.clock,
                )?
            };
            map.insert(col.name.clone(), value);
        }
        Ok(JsonValue::Object(map))
    }

    /// Decode a `StringArray`-encoded column for one row and invoke `f` with a
    /// borrowed `&[&str]` view over the decoded tokens.
    ///
    /// Why this exists: `decode_column_value` cannot return
    /// `ValueRef::StringArray(&[&str])` because the `&str` borrows would
    /// outlive the function. Callers that need per-token access must reach
    /// for an in-call closure form so the borrows stay scoped to one
    /// invocation. Tokens land in `scratch.strings` (`Vec<String>`), and the
    /// closure receives a freshly-built `Vec<&str>` view over them.
    ///
    /// Returns:
    ///   - `Ok(Some(R))` when the column was decoded and `f` was called.
    ///   - `Ok(None)` when the column tag was Null.
    ///   - `Err(...)` for malformed rowpack bytes or a non-StringArray
    ///     encoding at this column index.
    pub fn with_string_array_column<R>(
        &self,
        row: &[u8],
        col_idx: usize,
        scratch: &mut RowPackDecodeScratch,
        f: impl FnOnce(&[&str]) -> R,
    ) -> Result<Option<R>> {
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
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            let tag = tags[idx];
            if tag > 2 {
                return Err(anyhow!("rowpack invalid tag {}", tag));
            }
            if tag == 0 {
                if idx == col_idx {
                    return Ok(None);
                }
                continue;
            }
            if matches!(col.encoding, ColumnEncoding::Bool) {
                if idx == col_idx {
                    return Err(anyhow!("with_string_array_column called on Bool column"));
                }
                continue;
            }
            if idx == col_idx {
                let ColumnEncoding::StringArray { encoding } = &col.encoding else {
                    return Err(anyhow!(
                        "with_string_array_column called on non-StringArray column"
                    ));
                };
                let len = read_varint(&mut data)? as usize;
                if len > MAX_ARRAY_LEN {
                    return Err(anyhow!("rowpack string array length overflow"));
                }
                scratch.strings.clear();
                let dict = self.dict_values[idx].as_ref();
                for _ in 0..len {
                    if data.is_empty() {
                        return Err(anyhow!("rowpack string array token tag truncated"));
                    }
                    let token_tag = data[0];
                    data = &data[1..];
                    let token = if token_tag == 2 {
                        decode_string_fallback(&mut data)?
                    } else {
                        decode_string(&mut data, encoding, dict)?
                    };
                    scratch.strings.push(token);
                }
                let view: Vec<&str> = scratch.strings.iter().map(|s| s.as_str()).collect();
                return Ok(Some(f(&view)));
            }
            decode_value_skip(col, tag, &mut data, self.dict_values[idx].as_ref())?;
        }
        Err(anyhow!("rowpack column index out of range"))
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
                    self.clock,
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

        for &col_idx in col_indices {
            if col_idx >= self.ctx.columns.len() {
                return Err(anyhow!("column index out of bounds"));
            }
        }

        let mut requested_mask = vec![false; self.ctx.columns.len()];
        for &idx in col_indices {
            requested_mask[idx] = true;
        }

        // decode_row_to_builders walks columns in physical context order and
        // advances one builder for each requested column. Therefore builders
        // must be in context order, even when callers request an arbitrary
        // projection order such as [seed, target, timestamp]. We project back
        // to caller order after decoding.
        let mut decoded_indices = Vec::with_capacity(col_indices.len());
        let mut builders = Vec::with_capacity(col_indices.len());
        let mut fields = Vec::with_capacity(col_indices.len());
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            if !requested_mask[idx] {
                continue;
            }
            let (field, builder) =
                make_arrow_builder(&col.name, col.logical_type, rows.len(), self.clock)?;
            decoded_indices.push(idx);
            fields.push(field);
            builders.push(builder);
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
        let decoded = RecordBatch::try_new(Arc::new(schema), arrays)
            .map_err(|e: arrow_schema::ArrowError| anyhow!(e))?;

        if decoded_indices == col_indices {
            return Ok(decoded);
        }

        let mut by_context_index = std::collections::HashMap::with_capacity(decoded_indices.len());
        for (decoded_pos, context_idx) in decoded_indices.into_iter().enumerate() {
            by_context_index.insert(context_idx, decoded_pos);
        }
        let projection: Vec<usize> = col_indices
            .iter()
            .map(|idx| {
                by_context_index
                    .get(idx)
                    .copied()
                    .ok_or_else(|| anyhow!("decoded projection missing column {}", idx))
            })
            .collect::<Result<_>>()?;
        decoded
            .project(&projection)
            .map_err(|e: arrow_schema::ArrowError| anyhow!(e))
    }
}

#[inline]
fn float_storage_bytes(storage: FloatStorage) -> usize {
    match storage {
        FloatStorage::F16 => 2,
        FloatStorage::F32 => 4,
        FloatStorage::F64 => 8,
    }
}

fn read_float_array_len(
    data: &mut &[u8],
    storage: FloatStorage,
    fixed_len: Option<usize>,
    require_u32: bool,
) -> Result<usize> {
    let len = match fixed_len {
        Some(len) => len,
        None => read_varint(data)? as usize,
    };
    let elem_bytes = float_storage_bytes(storage);
    let max_len = data.len() / elem_bytes;
    if len > MAX_ARRAY_LEN || len > max_len || (require_u32 && len > u32::MAX as usize) {
        return Err(anyhow!("rowpack array length overflow"));
    }
    Ok(len)
}

fn read_int_array_len(data: &mut &[u8], fixed_len: Option<usize>, label: &str) -> Result<usize> {
    let len = match fixed_len {
        Some(len) => len,
        None => read_varint(data)? as usize,
    };
    if len > MAX_ARRAY_LEN {
        return Err(anyhow!("rowpack {label} array length overflow"));
    }
    Ok(len)
}

fn decode_value_msgpack(
    out: &mut Vec<u8>,
    col: &ColumnContext,
    tag: u8,
    data: &mut &[u8],
    dict: Option<&Vec<Vec<u8>>>,
    clock: TimeUnit,
) -> Result<()> {
    match &col.encoding {
        ColumnEncoding::IntDict {
            values,
            signed,
            varint,
            bits,
        } => {
            let v = decode_int_dict(values, *signed, *varint, *bits, tag, data)?;
            if *signed {
                mp::write_sint(out, v)?;
            } else {
                mp::write_uint(out, v as u64)?;
            }
        }
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
        ColumnEncoding::FloatDict { values, storage } => {
            let v = decode_float_dict(values, *storage, tag, data)?;
            match col.logical_type {
                IngestedType::Float32 => mp::write_f32(out, v as f32)?,
                _ => mp::write_f64(out, v)?,
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
            let (secs, nanos) = split_parts(zigzag_decode(read_varint(data)?), clock);
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
        ColumnEncoding::FloatArray { storage }
        | ColumnEncoding::FixedFloatArray { storage, .. } => {
            let fixed_len = float_array_encoding(&col.encoding).and_then(|(_, len)| len);
            let len = read_float_array_len(data, *storage, fixed_len, true)?;
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
        ColumnEncoding::Int8Array
        | ColumnEncoding::Int16Array
        | ColumnEncoding::Int32Array
        | ColumnEncoding::Int64Array
        | ColumnEncoding::FixedIntArray { .. } => {
            // Wire format is shared across signed widths: width is carried in
            // the ColumnEncoding tag, not in the per-element bytes. Decoded
            // values come back as i64 from zigzag-varint and write as
            // msgpack signed ints regardless of declared width.
            let len = read_int_array_len(data, int_array_fixed_len(&col.encoding), "int")?;
            mp::write_array_len(out, len as u32)?;
            for _ in 0..len {
                let v = zigzag_decode(read_varint(data)?);
                mp::write_sint(out, v)?;
            }
        }
        ColumnEncoding::StringArray { encoding } => {
            let len = read_varint(data)? as usize;
            if len > MAX_ARRAY_LEN {
                return Err(anyhow!("rowpack string array length overflow"));
            }
            mp::write_array_len(out, len as u32)?;
            for _ in 0..len {
                if data.is_empty() {
                    return Err(anyhow!("rowpack string array token tag truncated"));
                }
                let token_tag = data[0];
                *data = &data[1..];
                let value = if token_tag == 2 {
                    decode_string_fallback(data)?
                } else {
                    decode_string(data, encoding, dict)?
                };
                mp::write_str(out, &value)?;
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
    clock: TimeUnit,
) -> Result<JsonValue> {
    let value = match &col.encoding {
        ColumnEncoding::IntDict {
            values,
            signed,
            varint,
            bits,
        } => {
            let v = decode_int_dict(values, *signed, *varint, *bits, tag, data)?;
            if *signed {
                JsonValue::from(v)
            } else {
                JsonValue::from(v as u64)
            }
        }
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
        ColumnEncoding::FloatDict { values, storage } => {
            let v = decode_float_dict(values, *storage, tag, data)?;
            match col.logical_type {
                IngestedType::Float32 => JsonValue::from(v as f32),
                _ => JsonValue::from(v),
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
            let (secs, nanos) = split_parts(zigzag_decode(read_varint(data)?), clock);
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
        ColumnEncoding::FloatArray { storage }
        | ColumnEncoding::FixedFloatArray { storage, .. } => {
            let fixed_len = float_array_encoding(&col.encoding).and_then(|(_, len)| len);
            let len = read_float_array_len(data, *storage, fixed_len, false)?;
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
        ColumnEncoding::Int8Array
        | ColumnEncoding::Int16Array
        | ColumnEncoding::Int32Array
        | ColumnEncoding::Int64Array
        | ColumnEncoding::FixedIntArray { .. } => {
            let len = read_int_array_len(data, int_array_fixed_len(&col.encoding), "int")?;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let v = zigzag_decode(read_varint(data)?);
                values.push(JsonValue::from(v));
            }
            JsonValue::from(values)
        }
        ColumnEncoding::StringArray { encoding } => {
            let len = read_varint(data)? as usize;
            if len > MAX_ARRAY_LEN {
                return Err(anyhow!("rowpack string array length overflow"));
            }
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                if data.is_empty() {
                    return Err(anyhow!("rowpack string array token tag truncated"));
                }
                let token_tag = data[0];
                *data = &data[1..];
                let value = if token_tag == 2 {
                    decode_string_fallback(data)?
                } else {
                    decode_string(data, encoding, dict)?
                };
                values.push(JsonValue::from(value));
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
        ColumnEncoding::IntDict {
            values,
            signed,
            varint,
            bits,
        } => {
            let v = decode_int_dict(values, *signed, *varint, *bits, tag, data)?;
            if *signed {
                Ok(ValueRef::Int64(v))
            } else {
                Ok(ValueRef::UInt64(v as u64))
            }
        }
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
        ColumnEncoding::FloatDict { values, storage } => {
            let v = decode_float_dict(values, *storage, tag, data)?;
            match col.logical_type {
                IngestedType::Float32 => Ok(ValueRef::Float32(v as f32)),
                _ => Ok(ValueRef::Float64(v)),
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
            Ok(ValueRef::Timestamp(secs))
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
        ColumnEncoding::FloatArray { storage }
        | ColumnEncoding::FixedFloatArray { storage, .. } => {
            let fixed_len = float_array_encoding(&col.encoding).and_then(|(_, len)| len);
            let len = read_float_array_len(data, *storage, fixed_len, false)?;
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
        ColumnEncoding::Int8Array | ColumnEncoding::FixedIntArray { .. }
            if col.logical_type == IngestedType::Int8Array =>
        {
            let len = read_int_array_len(data, int_array_fixed_len(&col.encoding), "int8")?;
            scratch.int8.clear();
            for _ in 0..len {
                let v = zigzag_decode(read_varint(data)?);
                scratch.int8.push(v as i8);
            }
            Ok(ValueRef::Int8Array(&scratch.int8))
        }
        ColumnEncoding::Int16Array | ColumnEncoding::FixedIntArray { .. }
            if col.logical_type == IngestedType::Int16Array =>
        {
            let len = read_int_array_len(data, int_array_fixed_len(&col.encoding), "int16")?;
            scratch.int16.clear();
            for _ in 0..len {
                let v = zigzag_decode(read_varint(data)?);
                scratch.int16.push(v as i16);
            }
            Ok(ValueRef::Int16Array(&scratch.int16))
        }
        ColumnEncoding::Int32Array | ColumnEncoding::FixedIntArray { .. }
            if col.logical_type == IngestedType::Int32Array =>
        {
            let len = read_int_array_len(data, int_array_fixed_len(&col.encoding), "int32")?;
            scratch.int32.clear();
            for _ in 0..len {
                let v = zigzag_decode(read_varint(data)?);
                scratch.int32.push(v as i32);
            }
            Ok(ValueRef::Int32Array(&scratch.int32))
        }
        ColumnEncoding::Int64Array | ColumnEncoding::FixedIntArray { .. }
            if col.logical_type == IngestedType::Int64Array =>
        {
            let len = read_int_array_len(data, int_array_fixed_len(&col.encoding), "int64")?;
            scratch.int64.clear();
            for _ in 0..len {
                let v = zigzag_decode(read_varint(data)?);
                scratch.int64.push(v);
            }
            Ok(ValueRef::Int64Array(&scratch.int64))
        }
        ColumnEncoding::Int8Array
        | ColumnEncoding::Int16Array
        | ColumnEncoding::Int32Array
        | ColumnEncoding::Int64Array
        | ColumnEncoding::FixedIntArray { .. } => Err(anyhow!("rowpack invalid array type")),
        ColumnEncoding::StringArray { encoding } => {
            // Decode tokens into scratch.strings, advancing `data` past the payload.
            let len = read_varint(data)? as usize;
            if len > MAX_ARRAY_LEN {
                return Err(anyhow!("rowpack string array length overflow"));
            }
            scratch.strings.clear();
            for _ in 0..len {
                if data.is_empty() {
                    return Err(anyhow!("rowpack string array token tag truncated"));
                }
                let token_tag = data[0];
                *data = &data[1..];
                let token = if token_tag == 2 {
                    decode_string_fallback(data)?
                } else {
                    decode_string(data, encoding, dict)?
                };
                scratch.strings.push(token);
            }
            // StringArray cannot be returned as ValueRef::StringArray(&[&str]) from
            // this function because &str elements would borrow from scratch.strings
            // (a Vec<String>), but Rust cannot express that those borrows are valid
            // for the caller's lifetime without unsafe code.
            //
            // The data pointer has been advanced past the payload above (correct), so
            // subsequent columns in the same row decode cleanly.
            //
            // Callers must use decode_row_to_builders, which writes tokens directly
            // into an Arrow ListBuilder that owns its memory — no borrowed references.
            // If a zero-copy hot path is ever needed, consider a bump/arena allocator
            // over scratch memory so all tokens sit in one contiguous slab.
            Err(anyhow!(
                "rowpack: StringArray columns must be decoded via decode_row_to_builders; \
                 decode_column_value cannot return a borrowed &[&str] safely (see knowledge/rust/ownership-and-references.md)"
            ))
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
        ColumnEncoding::IntDict {
            values,
            signed,
            varint,
            bits,
        } => {
            let _ = decode_int_dict(values, *signed, *varint, *bits, tag, data)?;
        }
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
        ColumnEncoding::FloatDict { values, storage } => {
            let _ = decode_float_dict(values, *storage, tag, data)?;
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
        ColumnEncoding::String { encoding } => {
            if tag == 2 {
                // Inline fallback: [varint_len][raw_bytes] for all string encodings.
                // Matches decode_value_ref which calls decode_string_fallback_into.
                let len = read_varint(data)? as usize;
                if len > MAX_FIELD_BYTES {
                    return Err(anyhow!("rowpack string fallback too large"));
                }
                if data.len() < len {
                    return Err(anyhow!("rowpack string fallback truncated"));
                }
                *data = &data[len..];
            } else {
                match encoding {
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
                }
            }
        }
        ColumnEncoding::FloatArray { storage }
        | ColumnEncoding::FixedFloatArray { storage, .. } => {
            let fixed_len = float_array_encoding(&col.encoding).and_then(|(_, len)| len);
            let len = read_float_array_len(data, *storage, fixed_len, false)?;
            let total = len.saturating_mul(float_storage_bytes(*storage));
            if data.len() < total {
                return Err(anyhow!("rowpack array truncated"));
            }
            *data = &data[total..];
        }
        ColumnEncoding::Int8Array
        | ColumnEncoding::Int16Array
        | ColumnEncoding::Int32Array
        | ColumnEncoding::Int64Array
        | ColumnEncoding::FixedIntArray { .. } => {
            let len = read_int_array_len(data, int_array_fixed_len(&col.encoding), "int")?;
            for _ in 0..len {
                let _ = read_varint(data)?;
            }
        }
        ColumnEncoding::StringArray { encoding } => {
            let len = read_varint(data)? as usize;
            if len > MAX_ARRAY_LEN {
                return Err(anyhow!("rowpack string array length overflow"));
            }
            for _ in 0..len {
                if data.is_empty() {
                    return Err(anyhow!("rowpack string array token tag truncated"));
                }
                let token_tag = data[0];
                *data = &data[1..];
                // Decode and discard the token value, advancing `data` correctly.
                if token_tag == 2 {
                    let _ = decode_string_fallback(data)?;
                } else {
                    let _ = decode_string(data, encoding, dict)?;
                }
            }
        }
        ColumnEncoding::Bool => {}
    }
    Ok(())
}

/// When `signed` is false, `v` is an unsigned value reinterpreted as i64.
fn append_int_to_builder(
    builder: &mut dyn arrow_array::builder::ArrayBuilder,
    logical: IngestedType,
    signed: bool,
    v: i64,
) -> Result<()> {
    use arrow_array::builder::*;
    if signed {
        match logical {
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
        let v = v as u64;
        match logical {
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
    Ok(())
}

fn decode_int_dict(
    values: &[i64],
    signed: bool,
    varint: bool,
    bits: u8,
    tag: u8,
    data: &mut &[u8],
) -> Result<i64> {
    if tag == 1 {
        let idx = read_varint(data)? as usize;
        return values
            .get(idx)
            .copied()
            .ok_or_else(|| anyhow!("rowpack int dict index {idx} out of range"));
    }
    if signed {
        decode_int(data, varint, bits)
    } else {
        Ok(decode_uint(data, varint, bits)? as i64)
    }
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
        .downcast_mut::<TimestampSecondBuilder>()
    {
        b.append_null();
        return;
    }
    if let Some(b) = builder
        .as_any_mut()
        .downcast_mut::<TimestampMillisecondBuilder>()
    {
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
    if let Some(b) = builder
        .as_any_mut()
        .downcast_mut::<TimestampNanosecondBuilder>()
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
        return;
    }
    if let Some(b) = builder
        .as_any_mut()
        .downcast_mut::<ListBuilder<Int64Builder>>()
    {
        b.append_null();
        return;
    }
    if let Some(b) = builder
        .as_any_mut()
        .downcast_mut::<ListBuilder<StringBuilder>>()
    {
        b.append_null();
    }
}

pub fn make_arrow_builder(
    name: &str,
    dtype: IngestedType,
    capacity: usize,
    clock: TimeUnit,
) -> Result<(
    arrow_schema::Field,
    Box<dyn arrow_array::builder::ArrayBuilder>,
)> {
    use arrow_array::builder::*;
    use arrow_schema::DataType;

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
        IngestedType::Timestamp => match clock {
            TimeUnit::Nanoseconds => (
                DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
                Box::new(TimestampNanosecondBuilder::with_capacity(capacity)),
            ),
            TimeUnit::Microseconds => (
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
                Box::new(TimestampMicrosecondBuilder::with_capacity(capacity)),
            ),
            TimeUnit::Milliseconds => (
                DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, None),
                Box::new(TimestampMillisecondBuilder::with_capacity(capacity)),
            ),
            TimeUnit::Seconds | TimeUnit::Hours | TimeUnit::Days => (
                DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
                Box::new(TimestampSecondBuilder::with_capacity(capacity)),
            ),
            other => unreachable!("{other:?} is not an index clock"),
        },
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
        IngestedType::Int8Array => {
            let values_builder = Int8Builder::with_capacity(capacity * 32);
            (
                DataType::new_list(DataType::Int8, true),
                Box::new(ListBuilder::new(values_builder)),
            )
        }
        IngestedType::Int16Array => {
            let values_builder = Int16Builder::with_capacity(capacity * 32);
            (
                DataType::new_list(DataType::Int16, true),
                Box::new(ListBuilder::new(values_builder)),
            )
        }
        IngestedType::Int32Array => {
            let values_builder = Int32Builder::with_capacity(capacity * 32);
            (
                DataType::new_list(DataType::Int32, true),
                Box::new(ListBuilder::new(values_builder)),
            )
        }
        IngestedType::Int64Array => {
            let values_builder = Int64Builder::with_capacity(capacity * 32);
            (
                DataType::new_list(DataType::Int64, true),
                Box::new(ListBuilder::new(values_builder)),
            )
        }
        IngestedType::StringArray => {
            let values_builder = StringBuilder::with_capacity(capacity * 32, capacity * 128);
            (
                DataType::new_list(DataType::Utf8, true),
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
    clock: TimeUnit,
) -> Result<()> {
    use arrow_array::builder::*;

    // Helper to get typed builder. We trust the builder matches the column type because we created it that way.
    // However, we should be careful.

    match &col.encoding {
        ColumnEncoding::IntDict {
            values,
            signed,
            varint,
            bits,
        } => {
            let v = decode_int_dict(values, *signed, *varint, *bits, tag, data)?;
            append_int_to_builder(builder, col.logical_type, *signed, v)?;
        }
        ColumnEncoding::Int {
            signed,
            varint,
            bits,
        } => {
            let v = if *signed {
                decode_int(data, *varint, *bits)?
            } else {
                decode_uint(data, *varint, *bits)? as i64
            };
            append_int_to_builder(builder, col.logical_type, *signed, v)?;
        }
        ColumnEncoding::FloatDict { values, storage } => {
            let v = decode_float_dict(values, *storage, tag, data)?;
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
            let ticks = zigzag_decode(read_varint(data)?);
            let builder = builder.as_any_mut();
            match clock {
                TimeUnit::Nanoseconds => builder
                    .downcast_mut::<TimestampNanosecondBuilder>()
                    .unwrap()
                    .append_value(ticks),
                TimeUnit::Microseconds => builder
                    .downcast_mut::<TimestampMicrosecondBuilder>()
                    .unwrap()
                    .append_value(ticks),
                TimeUnit::Milliseconds => builder
                    .downcast_mut::<TimestampMillisecondBuilder>()
                    .unwrap()
                    .append_value(ticks),
                TimeUnit::Seconds | TimeUnit::Hours | TimeUnit::Days => builder
                    .downcast_mut::<TimestampSecondBuilder>()
                    .unwrap()
                    .append_value(convert_trunc(ticks, clock, TimeUnit::Seconds)),
                other => unreachable!("{other:?} is not an index clock"),
            }
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
        ColumnEncoding::FloatArray { storage }
        | ColumnEncoding::FixedFloatArray { storage, .. } => {
            let fixed_len = float_array_encoding(&col.encoding).and_then(|(_, len)| len);
            let len = read_float_array_len(data, *storage, fixed_len, false)?;
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
        ColumnEncoding::Int8Array | ColumnEncoding::FixedIntArray { .. }
            if col.logical_type == IngestedType::Int8Array =>
        {
            let len = read_int_array_len(data, int_array_fixed_len(&col.encoding), "int8")?;
            scratch.int8.clear();
            for _ in 0..len {
                let v = zigzag_decode(read_varint(data)?);
                scratch.int8.push(v as i8);
            }
            let b = builder
                .as_any_mut()
                .downcast_mut::<ListBuilder<Int8Builder>>()
                .unwrap();
            b.values().append_slice(&scratch.int8);
            b.append(true);
        }
        ColumnEncoding::Int16Array | ColumnEncoding::FixedIntArray { .. }
            if col.logical_type == IngestedType::Int16Array =>
        {
            let len = read_int_array_len(data, int_array_fixed_len(&col.encoding), "int16")?;
            scratch.int16.clear();
            for _ in 0..len {
                let v = zigzag_decode(read_varint(data)?);
                scratch.int16.push(v as i16);
            }
            let b = builder
                .as_any_mut()
                .downcast_mut::<ListBuilder<Int16Builder>>()
                .unwrap();
            b.values().append_slice(&scratch.int16);
            b.append(true);
        }
        ColumnEncoding::Int32Array | ColumnEncoding::FixedIntArray { .. }
            if col.logical_type == IngestedType::Int32Array =>
        {
            let len = read_int_array_len(data, int_array_fixed_len(&col.encoding), "int32")?;
            scratch.int32.clear();
            for _ in 0..len {
                let v = zigzag_decode(read_varint(data)?);
                scratch.int32.push(v as i32);
            }
            let b = builder
                .as_any_mut()
                .downcast_mut::<ListBuilder<Int32Builder>>()
                .unwrap();
            b.values().append_slice(&scratch.int32);
            b.append(true);
        }
        ColumnEncoding::Int64Array | ColumnEncoding::FixedIntArray { .. }
            if col.logical_type == IngestedType::Int64Array =>
        {
            let len = read_int_array_len(data, int_array_fixed_len(&col.encoding), "int64")?;
            scratch.int64.clear();
            for _ in 0..len {
                let v = zigzag_decode(read_varint(data)?);
                scratch.int64.push(v);
            }
            let b = builder
                .as_any_mut()
                .downcast_mut::<ListBuilder<Int64Builder>>()
                .unwrap();
            b.values().append_slice(&scratch.int64);
            b.append(true);
        }
        ColumnEncoding::Int8Array
        | ColumnEncoding::Int16Array
        | ColumnEncoding::Int32Array
        | ColumnEncoding::Int64Array
        | ColumnEncoding::FixedIntArray { .. } => {
            return Err(anyhow!("rowpack invalid array type"))
        }
        ColumnEncoding::StringArray { encoding } => {
            let len = read_varint(data)? as usize;
            if len > MAX_ARRAY_LEN {
                return Err(anyhow!("rowpack string array length overflow"));
            }
            let b = builder
                .as_any_mut()
                .downcast_mut::<ListBuilder<StringBuilder>>()
                .unwrap();
            for _ in 0..len {
                if data.is_empty() {
                    return Err(anyhow!("rowpack string array token tag truncated"));
                }
                let token_tag = data[0];
                *data = &data[1..];
                scratch.string.clear();
                if token_tag == 2 {
                    decode_string_fallback_into(data, &mut scratch.string)?;
                } else {
                    decode_string_into(data, encoding, dict, scratch)?;
                }
                b.values().append_value(&scratch.string);
            }
            b.append(true);
        }
        ColumnEncoding::Bool => return Err(anyhow!("rowpack bool unexpected")),
    }
    Ok(())
}
