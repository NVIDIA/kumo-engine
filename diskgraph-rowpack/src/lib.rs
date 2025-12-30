use anyhow::{anyhow, Result};
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use half::f16;
use base64::Engine;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use num_traits::Signed;
use rmp::encode as mp;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use ahash::RandomState;
use hashbrown::HashMap;

const ROWPACK_VERSION: u8 = 2;
const MAX_FIELD_BYTES: usize = 8 * 1024 * 1024;
const MAX_ARRAY_LEN: usize = 1_000_000;
const ALPHABET_MAX_CHARS: usize = 32;
const ALPHABET_MIN_COVERAGE: f32 = 0.95;
const NUMERIC_MIN_COVERAGE: f32 = 0.98;
const MAX_NUMERIC_SCALE: u32 = 38;

type FastHashMap<K, V> = HashMap<K, V, RandomState>;

#[derive(Clone)]
struct AlphabetMap {
    map: [i16; 256],
    bits: u8,
}

struct BpeTrie {
    next: Vec<[i32; 256]>,
    value: Vec<Option<u16>>,
}

impl BpeTrie {
    fn from_dict(dict: &[Vec<u8>]) -> Self {
        let mut next = Vec::new();
        let mut value = Vec::new();
        next.push([-1; 256]);
        value.push(None);
        for (idx, entry) in dict.iter().enumerate() {
            let mut node = 0usize;
            for &b in entry {
                let next_idx = next[node][b as usize];
                if next_idx < 0 {
                    next.push([-1; 256]);
                    value.push(None);
                    let new_idx = (next.len() - 1) as i32;
                    next[node][b as usize] = new_idx;
                    node = new_idx as usize;
                } else {
                    node = next_idx as usize;
                }
            }
            value[node] = Some(256u16 + idx as u16);
        }
        Self { next, value }
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
    #[serde(default)]
    pub string_iso_threshold: Option<f32>,
    #[serde(default)]
    pub enable_float16: Option<bool>,
    #[serde(default)]
    pub force_float64: Option<bool>,
    #[serde(default)]
    pub decimal_allow_lossy: Option<bool>,
}

impl Default for RowPackConfig {
    fn default() -> Self {
        Self {
            sample_rows: Some(10_000),
            max_dict_bytes_per_col: Some(1_000_000),
            max_dict_entries: Some(50_000),
            bpe_max_dict_bytes: Some(1_000_000),
            bpe_max_dict_entries: Some(4096),
            bpe_max_samples: Some(5_000),
            bpe_min_gain_ratio: Some(0.95),
            bpe_min_avg_len: Some(12),
            string_dict_coverage: Some(0.7),
            string_long_min_len: Some(128),
            string_long_min_ratio: Some(0.85),
            string_iso_threshold: Some(0.85),
            enable_float16: Some(true),
            force_float64: Some(false),
            decimal_allow_lossy: Some(true),
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
                string_iso_threshold: inner.string_iso_threshold.or(base.string_iso_threshold),
                enable_float16: inner.enable_float16.or(base.enable_float16),
                force_float64: inner.force_float64.or(base.force_float64),
                decimal_allow_lossy: inner.decimal_allow_lossy.or(base.decimal_allow_lossy),
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

    fn string_iso_threshold(&self) -> f32 {
        self.string_iso_threshold.unwrap_or(0.85)
    }

    fn enable_float16(&self) -> bool {
        self.enable_float16.unwrap_or(true)
    }

    fn force_float64(&self) -> bool {
        self.force_float64.unwrap_or(false)
    }

    fn decimal_allow_lossy(&self) -> bool {
        self.decimal_allow_lossy.unwrap_or(true)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogicalType {
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
    Date,
    Timestamp,
    Decimal128,
    Decimal256,
    Float32Array,
    Float64Array,
}

#[derive(Debug, Clone)]
pub struct RowPackColumn {
    pub name: String,
    pub logical_type: LogicalType,
    pub fixed_len: Option<usize>,
    pub precision: Option<u32>,
    pub scale: Option<i32>,
}

pub trait RowValueAccess {
    fn get_int64(&mut self, idx: usize) -> Result<Option<i64>>;
    fn get_int32(&mut self, idx: usize) -> Result<Option<i32>>;
    fn get_uint64(&mut self, idx: usize) -> Result<Option<u64>>;
    fn get_uint32(&mut self, idx: usize) -> Result<Option<u32>>;
    fn get_float64(&mut self, idx: usize) -> Result<Option<f64>>;
    fn get_float32(&mut self, idx: usize) -> Result<Option<f32>>;
    fn get_bool(&mut self, idx: usize) -> Result<Option<bool>>;
    fn get_string(&mut self, idx: usize) -> Result<Option<&str>>;
    fn get_bytes(&mut self, idx: usize) -> Result<Option<&[u8]>>;
    fn get_date(&mut self, idx: usize) -> Result<Option<i32>>;
    fn get_timestamp(&mut self, idx: usize) -> Result<Option<(i64, u32)>>;
    fn get_decimal128(&mut self, idx: usize) -> Result<Option<i128>>;
    fn get_decimal256(&mut self, idx: usize) -> Result<Option<[u8; 32]>>;
    fn get_float32_array(&mut self, idx: usize) -> Result<Option<&[f32]>>;
    fn get_float64_array(&mut self, idx: usize) -> Result<Option<&[f64]>>;
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
    pub logical_type: LogicalType,
    pub encoding: ColumnEncoding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColumnEncoding {
    Int { signed: bool, varint: bool, bits: u8 },
    Float { storage: FloatStorage },
    Bool,
    String { encoding: StringEncoding },
    Bytes { fixed_len: Option<usize> },
    Date { varint: bool, bits: u8 },
    Timestamp,
    Decimal {
        precision: u32,
        scale: i32,
        bytes: u8,
        storage: DecimalStorage,
    },
    FloatArray { storage: FloatStorage },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecimalStorage {
    Raw,
    Int { varint: bool, bits: u8 },
    Float { storage: FloatStorage },
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
    NumericInt,
    NumericScaled,
    Hex { len: usize, dashed: bool, upper: bool },
    IsoTimestamp { fmt: IsoFormat },
    Snappy,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IsoFormat {
    Rfc3339,
    DateOnly,
    DateTimeSpace,
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

fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ (-((value & 1) as i64))
}

fn encode_hex(value: &str, len: usize, dashed: bool) -> Option<Vec<u8>> {
    let raw: String = if dashed {
        value.chars().filter(|c| *c != '-').collect()
    } else {
        value.to_string()
    };
    if raw.len() != len {
        return None;
    }
    let mut out = Vec::with_capacity(len / 2);
    let mut chars = raw.chars();
    while let (Some(a), Some(b)) = (chars.next(), chars.next()) {
        let hi = a.to_digit(16)? as u8;
        let lo = b.to_digit(16)? as u8;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

fn decode_hex(bytes: &[u8], len: usize, dashed: bool, upper: bool) -> String {
    let mut out = String::with_capacity(len + if dashed { 4 } else { 0 });
    for (idx, byte) in bytes.iter().enumerate() {
        if dashed && (idx == 4 || idx == 6 || idx == 8 || idx == 10) {
            out.push('-');
        }
        if upper {
            out.push_str(&format!("{:02X}", byte));
        } else {
            out.push_str(&format!("{:02x}", byte));
        }
    }
    out
}

fn parse_iso(value: &str) -> Option<(IsoFormat, i64, u32)> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
        let secs = dt.timestamp();
        let nanos = dt.timestamp_subsec_nanos();
        return Some((IsoFormat::Rfc3339, secs, nanos));
    }
    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        let dt = date.and_hms_opt(0, 0, 0)?;
        return Some((IsoFormat::DateOnly, dt.and_utc().timestamp(), 0));
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S") {
        return Some((IsoFormat::DateTimeSpace, dt.and_utc().timestamp(), 0));
    }
    None
}

fn format_iso(fmt: IsoFormat, secs: i64, nanos: u32) -> String {
    match fmt {
        IsoFormat::Rfc3339 => {
            let dt = DateTime::<Utc>::from_timestamp(secs, nanos)
                .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap());
            dt.to_rfc3339()
        }
        IsoFormat::DateOnly => {
            let dt = DateTime::<Utc>::from_timestamp(secs, nanos)
                .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap());
            dt.date_naive().format("%Y-%m-%d").to_string()
        }
        IsoFormat::DateTimeSpace => {
            let dt = DateTime::<Utc>::from_timestamp(secs, nanos)
                .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap());
            dt.naive_utc().format("%Y-%m-%d %H:%M:%S").to_string()
        }
    }
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

#[derive(Debug, Clone)]
pub struct RowPackContextBuilder {
    config: RowPackConfig,
    columns: Vec<LogicalType>,
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
    bytes: u8,
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
                LogicalType::Decimal128 => match (col.precision, col.scale) {
                    (Some(precision), Some(scale)) => Some(DecimalSpec {
                        precision,
                        scale,
                        bytes: 16,
                    }),
                    _ => None,
                },
                LogicalType::Decimal256 => match (col.precision, col.scale) {
                    (Some(precision), Some(scale)) => Some(DecimalSpec {
                        precision,
                        scale,
                        bytes: 32,
                    }),
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

    pub fn finish(self) -> Result<RowPackContext> {
        let mut columns = Vec::with_capacity(self.columns.len());
        let mut estimate = 0usize;
        let mut bools = 0usize;
        for (idx, logical_type) in self.columns.iter().enumerate() {
            let encoding = match (logical_type, self.decimal_specs[idx]) {
                (LogicalType::Decimal128, Some(spec)) | (LogicalType::Decimal256, Some(spec)) => {
                    if spec.precision == 0 {
                        return Err(anyhow!("rowpack decimal precision missing"));
                    }
                    let storage = choose_decimal_storage(&self.config, spec.precision, spec.scale)?;
                    ColumnEncoding::Decimal {
                        precision: spec.precision,
                        scale: spec.scale,
                        bytes: spec.bytes,
                        storage,
                    }
                }
                (LogicalType::Decimal128, None) | (LogicalType::Decimal256, None) => {
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
                varint_bytes_u64(avg as u64) + (varint_bytes_u64(256) * 4)
            }
            StringEncoding::NumericInt => varint_bytes(64),
            StringEncoding::NumericScaled => varint_bytes(64) + varint_bytes(16),
            StringEncoding::Hex { len, .. } => len / 2,
            StringEncoding::IsoTimestamp { .. } => varint_bytes(64) + 5,
            StringEncoding::Snappy => {
                let avg = avg_len(stats.string_bytes, stats.string_total);
                varint_bytes_u64(avg as u64) + avg
            }
            StringEncoding::Raw => {
                let avg = avg_len(stats.string_bytes, stats.string_total);
                varint_bytes_u64(avg as u64) + avg
            }
        },
        ColumnEncoding::Decimal { bytes, storage, scale, .. } => match storage {
            DecimalStorage::Raw => *bytes as usize,
            DecimalStorage::Int { varint, bits } => {
                if *scale == 0 {
                    if *varint {
                        varint_bytes(*bits)
                    } else {
                        ((*bits + 7) / 8) as usize
                    }
                } else {
                    *bytes as usize
                }
            }
            DecimalStorage::Float { storage } => match storage {
                FloatStorage::F16 => 2,
                FloatStorage::F32 => 4,
                FloatStorage::F64 => 8,
            },
        },
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

fn choose_decimal_storage(
    config: &RowPackConfig,
    precision: u32,
    scale: i32,
) -> Result<DecimalStorage> {
    if !config.decimal_allow_lossy() {
        return Ok(DecimalStorage::Raw);
    }
    if scale == 0 {
        if precision <= 9 {
            return Ok(DecimalStorage::Int {
                varint: true,
                bits: 32,
            });
        }
        if precision <= 18 {
            return Ok(DecimalStorage::Int {
                varint: false,
                bits: 64,
            });
        }
        return Ok(DecimalStorage::Raw);
    }
    Ok(DecimalStorage::Float {
        storage: FloatStorage::F64,
    })
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
    numeric_hits: usize,
    numeric_scaled_hits: usize,
    numeric_i64_hits: usize,
    numeric_max_abs: i64,
    numeric_max_scale: u32,
    string_bytes: usize,
    string_total: usize,
    bytes_total: usize,
    bytes_count: usize,
    string_long_bytes: usize,
    string_long_raw: usize,
    string_long_compressed: usize,
    string_iso_hits: usize,
    string_iso_total: usize,
    string_iso_rfc3339: usize,
    string_iso_date: usize,
    string_iso_space: usize,
    string_hex_len: Option<usize>,
    string_hex_dashed: bool,
    string_hex_upper: Option<bool>,
    string_hex_ok: bool,
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
            string_counts: FastHashMap::default(),
            string_max_len: 0,
            string_samples: Vec::new(),
            string_sample_bytes: 0,
            alphabet_bytes: [false; 256],
            alphabet_len: 0,
            alphabet_ok: true,
            alphabet_hits: 0,
            numeric_hits: 0,
            numeric_scaled_hits: 0,
            numeric_i64_hits: 0,
            numeric_max_abs: 0,
            numeric_max_scale: 0,
            string_bytes: 0,
            string_total: 0,
            bytes_total: 0,
            bytes_count: 0,
            string_long_bytes: 0,
            string_long_raw: 0,
            string_long_compressed: 0,
            string_iso_hits: 0,
            string_iso_total: 0,
            string_iso_rfc3339: 0,
            string_iso_date: 0,
            string_iso_space: 0,
            string_hex_len: None,
            string_hex_dashed: false,
            string_hex_upper: None,
            string_hex_ok: true,
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
                if let Some((value, scale)) = parse_numeric_i64(v) {
                    self.numeric_hits += 1;
                    self.numeric_i64_hits += 1;
                    if scale > 0 {
                        self.numeric_scaled_hits += 1;
                    }
                    let abs = value.abs();
                    if abs > self.numeric_max_abs {
                        self.numeric_max_abs = abs;
                    }
                    if scale > self.numeric_max_scale {
                        self.numeric_max_scale = scale;
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
                if let Some((fmt, _, _)) = parse_iso(v) {
                    self.string_iso_hits += 1;
                    match fmt {
                        IsoFormat::Rfc3339 => self.string_iso_rfc3339 += 1,
                        IsoFormat::DateOnly => self.string_iso_date += 1,
                        IsoFormat::DateTimeSpace => self.string_iso_space += 1,
                    }
                }
                self.string_iso_total += 1;
                if self.string_hex_ok {
                    let trimmed = v.trim();
                    let (len, dashed) = if trimmed.len() == 36 {
                        (32usize, true)
                    } else {
                        (trimmed.len(), false)
                    };
                    if trimmed.len() == 32 || trimmed.len() == 36 {
                        let mut has_lower = false;
                        let mut has_upper = false;
                        for ch in trimmed.chars() {
                            if ch.is_ascii_lowercase() {
                                has_lower = true;
                            } else if ch.is_ascii_uppercase() {
                                has_upper = true;
                            }
                        }
                        if has_lower && has_upper {
                            self.string_hex_ok = false;
                        }
                        if encode_hex(trimmed, len, dashed).is_some() {
                            if let Some(existing) = self.string_hex_len {
                                if existing != len {
                                    self.string_hex_ok = false;
                                }
                            } else {
                                self.string_hex_len = Some(len);
                                self.string_hex_dashed = dashed;
                                if has_lower || has_upper {
                                    self.string_hex_upper = Some(has_upper);
                                }
                            }
                        } else {
                            self.string_hex_ok = false;
                        }
                    } else {
                        self.string_hex_ok = false;
                    }
                }
                if bytes.len() >= config.string_long_min_len() {
                    self.string_long_bytes += bytes.len();
                    self.string_long_raw += bytes.len();
                    let compressed = snap::raw::Encoder::new().compress_vec(bytes)?;
                    self.string_long_compressed += compressed.len();
                }
            }
            ValueRef::Bytes(v) => {
                self.bytes_total += v.len();
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

    fn choose_encoding(
        &self,
        logical: LogicalType,
        config: &RowPackConfig,
        fixed_len: Option<usize>,
    ) -> Result<ColumnEncoding> {
        match logical {
            LogicalType::Int64 => Ok(ColumnEncoding::Int {
                signed: true,
                varint: int_use_varint(self.int_min, self.int_max),
                bits: 64,
            }),
            LogicalType::Int32 => Ok(ColumnEncoding::Int {
                signed: true,
                varint: int_use_varint(self.int_min, self.int_max),
                bits: 32,
            }),
            LogicalType::Int16 => Ok(ColumnEncoding::Int {
                signed: true,
                varint: int_use_varint(self.int_min, self.int_max),
                bits: 16,
            }),
            LogicalType::Int8 => Ok(ColumnEncoding::Int {
                signed: true,
                varint: false,
                bits: 8,
            }),
            LogicalType::UInt64 => Ok(ColumnEncoding::Int {
                signed: false,
                varint: uint_use_varint(self.uint_max),
                bits: 64,
            }),
            LogicalType::UInt32 => Ok(ColumnEncoding::Int {
                signed: false,
                varint: uint_use_varint(self.uint_max),
                bits: 32,
            }),
            LogicalType::UInt16 => Ok(ColumnEncoding::Int {
                signed: false,
                varint: uint_use_varint(self.uint_max),
                bits: 16,
            }),
            LogicalType::UInt8 => Ok(ColumnEncoding::Int {
                signed: false,
                varint: false,
                bits: 8,
            }),
            LogicalType::Float64 => Ok(ColumnEncoding::Float {
                storage: choose_float_storage(config, self.float_min, self.float_max, true),
            }),
            LogicalType::Float32 => Ok(ColumnEncoding::Float {
                storage: choose_float_storage(config, self.float_min, self.float_max, false),
            }),
            LogicalType::Bool => Ok(ColumnEncoding::Bool),
            LogicalType::String => Ok(ColumnEncoding::String {
                encoding: choose_string_encoding(self, config)?,
            }),
            LogicalType::Bytes => Ok(ColumnEncoding::Bytes { fixed_len }),
            LogicalType::Date => Ok(ColumnEncoding::Date {
                varint: int_use_varint(self.int_min, self.int_max),
                bits: 32,
            }),
            LogicalType::Timestamp => Ok(ColumnEncoding::Timestamp),
            LogicalType::Decimal128 | LogicalType::Decimal256 => {
                Err(anyhow!("rowpack decimal encoding missing"))
            }
            LogicalType::Float32Array => Ok(ColumnEncoding::FloatArray {
                storage: choose_float_storage(config, self.float_min, self.float_max, false),
            }),
            LogicalType::Float64Array => Ok(ColumnEncoding::FloatArray {
                storage: choose_float_storage(config, self.float_min, self.float_max, true),
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

fn choose_float_storage(config: &RowPackConfig, min: f64, max: f64, is_f64: bool) -> FloatStorage {
    if config.force_float64() {
        return FloatStorage::F64;
    }
    if config.enable_float16() && min >= 0.001 && max <= 1000.0 {
        return FloatStorage::F16;
    }
    if is_f64 {
        FloatStorage::F32
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
    let bpe_allowed =
        avg_len >= config.bpe_min_avg_len() && avg_len < config.string_long_min_len();
    let iso_ratio = stats.string_iso_hits as f32 / total;
    if stats.string_iso_total > 0 && iso_ratio >= config.string_iso_threshold() {
        let mut fmt = IsoFormat::Rfc3339;
        let mut best = stats.string_iso_rfc3339;
        if stats.string_iso_date > best {
            best = stats.string_iso_date;
            fmt = IsoFormat::DateOnly;
        }
        if stats.string_iso_space > best {
            fmt = IsoFormat::DateTimeSpace;
        }
        return Ok(StringEncoding::IsoTimestamp {
            fmt,
        });
    }
    if stats.string_hex_ok {
        if let Some(len) = stats.string_hex_len {
            return Ok(StringEncoding::Hex {
                len,
                dashed: stats.string_hex_dashed,
                upper: stats.string_hex_upper.unwrap_or(false),
            });
        }
    }
    if total > 0.0 {
        let numeric_ratio = stats.numeric_i64_hits as f32 / total;
        if numeric_ratio >= NUMERIC_MIN_COVERAGE {
            if stats.numeric_scaled_hits > 0 {
                return Ok(StringEncoding::NumericScaled);
            }
            return Ok(StringEncoding::NumericInt);
        }
    }
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
        if coverage_ratio >= config.string_dict_coverage()
            && avg_hits >= 2.0
            && !values.is_empty()
        {
            return Ok(StringEncoding::Dict { values });
        }
    }
    if stats.string_long_bytes > 0 {
        let ratio = stats.string_long_compressed as f32 / stats.string_long_raw as f32;
        if ratio <= config.string_long_min_ratio() {
            return Ok(StringEncoding::Snappy);
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
        let mut pair_counts: FastHashMap<(u16, u16), usize> = FastHashMap::default();
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
    dict_entries
}

pub struct RowPackEncoder {
    ctx: RowPackContext,
    dict_maps: Vec<Option<FastHashMap<Vec<u8>, u32>>>,
    alphabet_maps: Vec<Option<AlphabetMap>>,
    bpe_tries: Vec<Option<BpeTrie>>,
}

pub struct RowPackScratch {
    tags: BitWriter,
    bools: BitWriter,
    data: Vec<u8>,
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
            self.bools.buf.reserve(bool_bytes - self.bools.buf.capacity());
        }
        if self.data.capacity() < ctx.estimated_row_bytes as usize {
            self.data
                .reserve(ctx.estimated_row_bytes as usize - self.data.capacity());
        }
        self.tags.reset();
        self.bools.reset();
        self.data.clear();
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
                    let mut map = FastHashMap::with_capacity_and_hasher(
                        values.len(),
                        RandomState::default(),
                    );
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
                    let mut map = [-1i16; 256];
                    for (idx, b) in alphabet.iter().enumerate() {
                        map[*b as usize] = idx as i16;
                    }
                    alphabet_maps.push(Some(AlphabetMap {
                        map,
                        bits: alphabet_bits(alphabet.len()),
                    }));
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

    pub fn encode_row<'a>(&self, values: &[ValueRef<'a>]) -> Result<Vec<u8>> {
        if values.len() != self.ctx.columns.len() {
            return Err(anyhow!("rowpack column mismatch"));
        }
        let tag_bytes = (self.ctx.columns.len() * 2 + 7) / 8;
        let bool_columns = self
            .ctx
            .columns
            .iter()
            .filter(|c| matches!(c.encoding, ColumnEncoding::Bool))
            .count();
        let bool_bytes = (bool_columns + 7) / 8;
        let mut tags = BitWriter::with_capacity(tag_bytes);
        let mut bools = BitWriter::with_capacity(bool_bytes);
        let mut data = Vec::with_capacity(self.ctx.estimated_row_bytes as usize);
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            let value = &values[idx];
            let tag = match value {
                ValueRef::Null => 0u8,
                _ if matches!(col.encoding, ColumnEncoding::Bool) => {
                    let v = match value {
                        ValueRef::Bool(v) => *v,
                        _ => return Err(anyhow!("rowpack bool mismatch")),
                    };
                    bools.write_bits(if v { 1 } else { 0 }, 1);
                    1u8
                }
                _ => encode_value_with_tag(
                    &mut data,
                    col,
                    value,
                    self.dict_maps[idx].as_ref(),
                    self.alphabet_maps[idx].as_ref(),
                    self.bpe_tries[idx].as_ref(),
                )?,
            };
            tags.write_bits(tag, 2);
        }
        let mut out = tags.finish();
        let mut bool_bytes = bools.finish();
        out.append(&mut bool_bytes);
        out.extend_from_slice(&data);
        Ok(out)
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
                _ => encode_value_with_tag(
                    &mut scratch.data,
                    col,
                    value,
                    self.dict_maps[idx].as_ref(),
                    self.alphabet_maps[idx].as_ref(),
                    self.bpe_tries[idx].as_ref(),
                )?,
            };
            scratch.tags.write_bits(tag, 2);
        }
        out.clear();
        out.reserve(
            scratch.tags.buf.len() + scratch.bools.buf.len() + scratch.data.len() + 8,
        );
        scratch.tags.append_to(out);
        scratch.bools.append_to(out);
        out.extend_from_slice(&scratch.data);
        Ok(())
    }

    pub fn encode_row_from_access<A: RowValueAccess>(
        &self,
        access: &mut A,
        out: &mut Vec<u8>,
        scratch: &mut RowPackScratch,
    ) -> Result<()> {
        scratch.reset(&self.ctx);
        for (idx, col) in self.ctx.columns.iter().enumerate() {
            let tag = match col.logical_type {
                LogicalType::Bool => match access.get_bool(idx)? {
                    None => 0u8,
                    Some(v) => {
                        scratch.bools.write_bits(if v { 1 } else { 0 }, 1);
                        1u8
                    }
                },
                LogicalType::Int64 => match access.get_int64(idx)? {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int { signed, varint, bits } = &col.encoding {
                            encode_int(&mut scratch.data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack int64 encoding mismatch"));
                        }
                    }
                },
                LogicalType::Int16 => match access.get_int32(idx)? {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int { signed, varint, bits } = &col.encoding {
                            encode_int(&mut scratch.data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack int16 encoding mismatch"));
                        }
                    }
                },
                LogicalType::Int8 => match access.get_int32(idx)? {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int { signed, varint, bits } = &col.encoding {
                            encode_int(&mut scratch.data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack int8 encoding mismatch"));
                        }
                    }
                },
                LogicalType::Int32 => match access.get_int32(idx)? {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int { signed, varint, bits } = &col.encoding {
                            encode_int(&mut scratch.data, *signed, *varint, *bits, v as i128)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack int32 encoding mismatch"));
                        }
                    }
                },
                LogicalType::UInt64 => match access.get_uint64(idx)? {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int { signed, varint, bits } = &col.encoding {
                            encode_uint(&mut scratch.data, *signed, *varint, *bits, v)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack uint64 encoding mismatch"));
                        }
                    }
                },
                LogicalType::UInt16 => match access.get_uint32(idx)? {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int { signed, varint, bits } = &col.encoding {
                            encode_uint(&mut scratch.data, *signed, *varint, *bits, v as u64)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack uint16 encoding mismatch"));
                        }
                    }
                },
                LogicalType::UInt8 => match access.get_uint32(idx)? {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int { signed, varint, bits } = &col.encoding {
                            encode_uint(&mut scratch.data, *signed, *varint, *bits, v as u64)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack uint8 encoding mismatch"));
                        }
                    }
                },
                LogicalType::UInt32 => match access.get_uint32(idx)? {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Int { signed, varint, bits } = &col.encoding {
                            encode_uint(&mut scratch.data, *signed, *varint, *bits, v as u64)?;
                            1u8
                        } else {
                            return Err(anyhow!("rowpack uint32 encoding mismatch"));
                        }
                    }
                },
                LogicalType::Float64 => match access.get_float64(idx)? {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Float { storage } = &col.encoding {
                            encode_float_with_tag(&mut scratch.data, col.logical_type, *storage, v)?
                        } else {
                            return Err(anyhow!("rowpack float64 encoding mismatch"));
                        }
                    }
                },
                LogicalType::Float32 => match access.get_float32(idx)? {
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
                LogicalType::String => match access.get_string(idx)? {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::String { encoding } = &col.encoding {
                            encode_string_with_tag(
                                &mut scratch.data,
                                encoding,
                                v,
                                self.dict_maps[idx].as_ref(),
                                self.alphabet_maps[idx].as_ref(),
                                self.bpe_tries[idx].as_ref(),
                            )?
                        } else {
                            return Err(anyhow!("rowpack string encoding mismatch"));
                        }
                    }
                },
                LogicalType::Bytes => match access.get_bytes(idx)? {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Bytes { fixed_len } = &col.encoding {
                            if let Some(len) = fixed_len {
                                if *len == v.len() {
                                    scratch.data.extend_from_slice(v);
                                    1u8
                                } else {
                                    write_varint(v.len() as u64, &mut scratch.data);
                                    scratch.data.extend_from_slice(v);
                                    2u8
                                }
                            } else {
                                write_varint(v.len() as u64, &mut scratch.data);
                                scratch.data.extend_from_slice(v);
                                1u8
                            }
                        } else {
                            return Err(anyhow!("rowpack bytes encoding mismatch"));
                        }
                    }
                },
                LogicalType::Date => match access.get_date(idx)? {
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
                LogicalType::Timestamp => match access.get_timestamp(idx)? {
                    None => 0u8,
                    Some((secs, nanos)) => {
                        write_varint(zigzag_encode(secs), &mut scratch.data);
                        write_varint(nanos as u64, &mut scratch.data);
                        1u8
                    }
                },
                LogicalType::Decimal128 => match access.get_decimal128(idx)? {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Decimal { bytes, scale, storage, .. } =
                            &col.encoding
                        {
                            if *bytes != 16 {
                                return Err(anyhow!("rowpack decimal size mismatch"));
                            }
                            encode_decimal_i128_with_tag(&mut scratch.data, v, *scale, storage)?
                        } else {
                            return Err(anyhow!("rowpack decimal128 encoding mismatch"));
                        }
                    }
                },
                LogicalType::Decimal256 => match access.get_decimal256(idx)? {
                    None => 0u8,
                    Some(v) => {
                        if let ColumnEncoding::Decimal { bytes, scale, storage, .. } =
                            &col.encoding
                        {
                            if *bytes != 32 {
                                return Err(anyhow!("rowpack decimal size mismatch"));
                            }
                            encode_decimal_i256_with_tag(&mut scratch.data, &v, *scale, storage)?
                        } else {
                            return Err(anyhow!("rowpack decimal256 encoding mismatch"));
                        }
                    }
                },
                LogicalType::Float32Array => match access.get_float32_array(idx)? {
                    None => 0u8,
                    Some(values) => {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            encode_float_array_f32_with_tag(
                                &mut scratch.data,
                                col.logical_type,
                                *storage,
                                values,
                            )?
                        } else {
                            return Err(anyhow!("rowpack float32 array encoding mismatch"));
                        }
                    }
                },
                LogicalType::Float64Array => match access.get_float64_array(idx)? {
                    None => 0u8,
                    Some(values) => {
                        if let ColumnEncoding::FloatArray { storage } = &col.encoding {
                            encode_float_array_f64_with_tag(
                                &mut scratch.data,
                                col.logical_type,
                                *storage,
                                values,
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
        out.reserve(
            scratch.tags.buf.len() + scratch.bools.buf.len() + scratch.data.len() + 8,
        );
        scratch.tags.append_to(out);
        scratch.bools.append_to(out);
        out.extend_from_slice(&scratch.data);
        Ok(())
    }
}

fn encode_value_with_tag<'a>(
    out: &mut Vec<u8>,
    col: &ColumnContext,
    value: &ValueRef<'a>,
    dict: Option<&FastHashMap<Vec<u8>, u32>>,
    alphabet: Option<&AlphabetMap>,
    bpe: Option<&BpeTrie>,
 ) -> Result<u8> {
    match (&col.encoding, value) {
        (ColumnEncoding::Int { signed, varint, bits }, ValueRef::Int64(v)) => {
            encode_int(out, *signed, *varint, *bits, *v as i128)?;
            Ok(1)
        }
        (ColumnEncoding::Int { signed, varint, bits }, ValueRef::Int32(v)) => {
            encode_int(out, *signed, *varint, *bits, *v as i128)?;
            Ok(1)
        }
        (ColumnEncoding::Int { signed, varint, bits }, ValueRef::UInt64(v)) => {
            encode_uint(out, *signed, *varint, *bits, *v)?;
            Ok(1)
        }
        (ColumnEncoding::Int { signed, varint, bits }, ValueRef::UInt32(v)) => {
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
            if *bytes != 16 {
                return Err(anyhow!("rowpack decimal size mismatch"));
            }
            encode_decimal_i128_with_tag(out, *v, *scale, storage)
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
            if *bytes != 32 {
                return Err(anyhow!("rowpack decimal size mismatch"));
            }
            encode_decimal_i256_with_tag(out, v, *scale, storage)
        }
        (ColumnEncoding::String { encoding }, ValueRef::String(v)) => {
            encode_string_with_tag(out, encoding, v, dict, alphabet, bpe)
        }
        (ColumnEncoding::FloatArray { storage }, ValueRef::Float32Array(values)) => {
            let tag = encode_float_array_f32_with_tag(
                out,
                col.logical_type,
                *storage,
                *values,
            )?;
            Ok(tag)
        }
        (ColumnEncoding::FloatArray { storage }, ValueRef::Float64Array(values)) => {
            let tag = encode_float_array_f64_with_tag(
                out,
                col.logical_type,
                *storage,
                *values,
            )?;
            Ok(tag)
        }
        _ => Err(anyhow!("rowpack value type mismatch")),
    }
}

fn encode_float_with_tag(
    out: &mut Vec<u8>,
    logical: LogicalType,
    storage: FloatStorage,
    v: f64,
) -> Result<u8> {
    if storage == FloatStorage::F16 && (!v.is_finite() || v < 0.001 || v > 1000.0) {
        match logical {
            LogicalType::Float32 => out.extend_from_slice(&(v as f32).to_le_bytes()),
            _ => out.extend_from_slice(&v.to_le_bytes()),
        }
        return Ok(2);
    }
    encode_float(out, storage, v)?;
    Ok(1)
}

fn encode_decimal_i128_with_tag(
    out: &mut Vec<u8>,
    value: i128,
    scale: i32,
    storage: &DecimalStorage,
) -> Result<u8> {
    match storage {
        DecimalStorage::Raw => {
            out.extend_from_slice(&value.to_le_bytes());
            Ok(1)
        }
        DecimalStorage::Int { varint, bits } => {
            if scale != 0 {
                out.extend_from_slice(&value.to_le_bytes());
                return Ok(2);
            }
            if value < i64::MIN as i128 || value > i64::MAX as i128 {
                out.extend_from_slice(&value.to_le_bytes());
                return Ok(2);
            }
            encode_int(out, true, *varint, *bits, value)?;
            Ok(1)
        }
        DecimalStorage::Float { storage } => {
            let divisor = 10f64.powi(scale);
            let float_value = (value as f64) / divisor;
            if !float_value.is_finite() {
                out.extend_from_slice(&value.to_le_bytes());
                return Ok(2);
            }
            let _ = encode_float_with_tag(out, LogicalType::Float64, *storage, float_value)?;
            Ok(1)
        }
    }
}

fn encode_decimal_i256_with_tag(
    out: &mut Vec<u8>,
    value: &[u8; 32],
    scale: i32,
    storage: &DecimalStorage,
) -> Result<u8> {
    match storage {
        DecimalStorage::Raw => {
            out.extend_from_slice(value);
            Ok(1)
        }
        DecimalStorage::Int { varint, bits } => {
            if scale != 0 {
                out.extend_from_slice(value);
                return Ok(2);
            }
            let big = BigInt::from_signed_bytes_le(value);
            if let Some(v) = big.to_i64() {
                encode_int(out, true, *varint, *bits, v as i128)?;
                Ok(1)
            } else {
                out.extend_from_slice(value);
                Ok(2)
            }
        }
        DecimalStorage::Float { storage } => {
            let big = BigInt::from_signed_bytes_le(value);
            let float_value = match big.to_f64() {
                Some(v) => v / 10f64.powi(scale),
                None => {
                    out.extend_from_slice(value);
                    return Ok(2);
                }
            };
            if !float_value.is_finite() {
                out.extend_from_slice(value);
                return Ok(2);
            }
            let _ = encode_float_with_tag(out, LogicalType::Float64, *storage, float_value)?;
            Ok(1)
        }
    }
}

fn encode_float_array_f32_with_tag(
    out: &mut Vec<u8>,
    logical: LogicalType,
    storage: FloatStorage,
    values: &[f32],
) -> Result<u8> {
    if storage == FloatStorage::F16 {
        let mut ok = true;
        for v in values {
            let v = *v as f64;
            if !v.is_finite() || v < 0.001 || v > 1000.0 {
                ok = false;
                break;
            }
        }
        if !ok {
            write_varint(values.len() as u64, out);
            match logical {
                LogicalType::Float64Array => {
                    for v in values {
                        out.extend_from_slice(&(*v as f64).to_le_bytes());
                    }
                }
                _ => {
                    for v in values {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            return Ok(2);
        }
    }
    write_varint(values.len() as u64, out);
    for v in values {
        encode_float(out, storage, *v as f64)?;
    }
    Ok(1)
}

fn encode_float_array_f64_with_tag(
    out: &mut Vec<u8>,
    logical: LogicalType,
    storage: FloatStorage,
    values: &[f64],
) -> Result<u8> {
    if storage == FloatStorage::F16 {
        let mut ok = true;
        for v in values {
            if !v.is_finite() || *v < 0.001 || *v > 1000.0 {
                ok = false;
                break;
            }
        }
        if !ok {
            write_varint(values.len() as u64, out);
            match logical {
                LogicalType::Float32Array => {
                    for v in values {
                        out.extend_from_slice(&(*v as f32).to_le_bytes());
                    }
                }
                _ => {
                    for v in values {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            return Ok(2);
        }
    }
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

fn encode_string_with_tag(
    out: &mut Vec<u8>,
    encoding: &StringEncoding,
    value: &str,
    dict: Option<&FastHashMap<Vec<u8>, u32>>,
    alphabet: Option<&AlphabetMap>,
    bpe: Option<&BpeTrie>,
) -> Result<u8> {
    match encoding {
        StringEncoding::Raw => {
            let bytes = value.as_bytes();
            write_varint(bytes.len() as u64, out);
            out.extend_from_slice(bytes);
            Ok(1)
        }
        StringEncoding::Dict { .. } => {
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
        StringEncoding::Alphabet { .. } => {
            let bytes = value.as_bytes();
            let map = alphabet.ok_or_else(|| anyhow!("rowpack missing alphabet map"))?;
            for b in bytes {
                if map.map[*b as usize] < 0 {
                    write_varint(bytes.len() as u64, out);
                    out.extend_from_slice(bytes);
                    return Ok(2);
                }
            }
            write_varint(bytes.len() as u64, out);
            write_alphabet_packed(out, bytes, &map.map, map.bits);
            Ok(1)
        }
        StringEncoding::Bpe { .. } => {
            let bytes = value.as_bytes();
            let trie = bpe.ok_or_else(|| anyhow!("rowpack missing bpe trie"))?;
            let mut tokens = Vec::new();
            tokenize_bpe(bytes, trie, &mut tokens);
            write_varint(tokens.len() as u64, out);
            for token in tokens {
                write_varint(token as u64, out);
            }
            Ok(1)
        }
        StringEncoding::NumericInt => {
            if let Some((value, scale)) = parse_numeric_i64(value) {
                if scale == 0 {
                    write_varint(zigzag_encode(value), out);
                    return Ok(1);
                }
            }
            let bytes = value.as_bytes();
            write_varint(bytes.len() as u64, out);
            out.extend_from_slice(bytes);
            Ok(2)
        }
        StringEncoding::NumericScaled => {
            if let Some((value, scale)) = parse_numeric_i64(value) {
                write_varint(zigzag_encode(value), out);
                write_varint(scale as u64, out);
                return Ok(1);
            }
            let bytes = value.as_bytes();
            write_varint(bytes.len() as u64, out);
            out.extend_from_slice(bytes);
            Ok(2)
        }
        StringEncoding::Hex { len, dashed, .. } => {
            if let Some(bytes) = encode_hex(value, *len, *dashed) {
                out.extend_from_slice(&bytes);
                Ok(1)
            } else {
                let bytes = value.as_bytes();
                write_varint(bytes.len() as u64, out);
                out.extend_from_slice(bytes);
                Ok(2)
            }
        }
        StringEncoding::IsoTimestamp { .. } => {
            if let Some((_fmt, secs, nanos)) = parse_iso(value) {
                write_varint(zigzag_encode(secs), out);
                write_varint(nanos as u64, out);
                Ok(1)
            } else {
                let bytes = value.as_bytes();
                write_varint(bytes.len() as u64, out);
                out.extend_from_slice(bytes);
                Ok(2)
            }
        }
        StringEncoding::Snappy => {
            let bytes = value.as_bytes();
            let compressed = snap::raw::Encoder::new().compress_vec(bytes)?;
            if compressed.len() < bytes.len() {
                write_varint(compressed.len() as u64, out);
                out.extend_from_slice(&compressed);
                Ok(1)
            } else {
                write_varint(bytes.len() as u64, out);
                out.extend_from_slice(bytes);
                Ok(2)
            }
        }
    }
}

fn alphabet_bits(len: usize) -> u8 {
    if len <= 1 {
        1
    } else {
        (usize::BITS - (len - 1).leading_zeros()) as u8
    }
}

fn write_alphabet_packed(out: &mut Vec<u8>, bytes: &[u8], map: &[i16; 256], bits: u8) {
    let mut cur: u64 = 0;
    let mut used: u8 = 0;
    for b in bytes {
        let idx = map[*b as usize] as u64;
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
}

fn tokenize_bpe(bytes: &[u8], trie: &BpeTrie, tokens: &mut Vec<u32>) {
    tokens.clear();
    let mut i = 0usize;
    while i < bytes.len() {
        let mut node = 0usize;
        let mut j = i;
        let mut last = None;
        while j < bytes.len() {
            let next_idx = trie.next[node][bytes[j] as usize];
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
    let mut tokens = Vec::new();
    let mut total = 0u64;
    for sample in samples {
        tokenize_bpe(sample, &trie, &mut tokens);
        total += varint_bytes_u64(tokens.len() as u64) as u64;
        for token in &tokens {
            total += varint_bytes_u64(*token as u64) as u64;
        }
    }
    total
}

fn parse_numeric_i64(value: &str) -> Option<(i64, u32)> {
    if value.starts_with('.') || value.ends_with('.') {
        return None;
    }
    let mut chars = value.as_bytes().iter().copied().peekable();
    let mut negative = false;
    if let Some(b'-') = chars.peek().copied() {
        negative = true;
        let _ = chars.next();
    }
    let mut seen_digit = false;
    let mut scale = 0u32;
    let mut seen_dot = false;
    let mut acc: i128 = 0;
    while let Some(b) = chars.next() {
        if b == b'.' {
            if seen_dot {
                return None;
            }
            seen_dot = true;
            continue;
        }
        if !b.is_ascii_digit() {
            return None;
        }
        seen_digit = true;
        let digit = (b - b'0') as i128;
        acc = acc.checked_mul(10)?.checked_add(digit)?;
        if seen_dot {
            scale = scale.saturating_add(1);
        }
    }
    if !seen_digit {
        return None;
    }
    if scale > MAX_NUMERIC_SCALE {
        return None;
    }
    if negative {
        acc = -acc;
    }
    if acc < i64::MIN as i128 || acc > i64::MAX as i128 {
        return None;
    }
    Some((acc as i64, scale))
}

fn format_scaled_i64(value: i64, scale: u32) -> String {
    if scale == 0 {
        return value.to_string();
    }
    let negative = value < 0;
    let digits = value.abs().to_string();
    let scale = scale as usize;
    if digits.len() <= scale {
        let mut out = String::new();
        if negative {
            out.push('-');
        }
        out.push('0');
        out.push('.');
        for _ in 0..(scale - digits.len()) {
            out.push('0');
        }
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

pub struct RowPackDecoder {
    ctx: RowPackContext,
    dict_values: Vec<Option<Vec<Vec<u8>>>>,
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
                decode_value_json(
                    col,
                    tag,
                    &mut data,
                    self.dict_values[idx].as_ref(),
                )?
            };
            map.insert(col.name.clone(), value);
        }
        Ok(JsonValue::Object(map))
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
        ColumnEncoding::Int { signed, varint, bits } => {
            if *signed {
                let v = decode_int(data, *varint, *bits)?;
                mp::write_sint(out, v)?;
            } else {
                let v = decode_uint(data, *varint, *bits)?;
                mp::write_uint(out, v)?;
            }
        }
        ColumnEncoding::Float { storage } => {
            let v = if tag == 2 {
                decode_float_fallback(data, col.logical_type)?
            } else {
                decode_float(data, *storage)?
            };
            match col.logical_type {
                LogicalType::Float32 => mp::write_f32(out, v as f32)?,
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
                    DecimalStorage::Int { varint, bits } => {
                        let v = decode_int(data, *varint, *bits)?;
                        if *scale == 0 {
                            mp::write_sint(out, v)?;
                        } else {
                            let text = format_decimal_i64(v, *scale);
                            mp::write_str(out, &text)?;
                        }
                    }
                    DecimalStorage::Float { storage } => {
                        let v = decode_float(data, *storage)?;
                        mp::write_f64(out, v)?;
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
            let elem_bytes = if tag == 2 {
                match col.logical_type {
                    LogicalType::Float32Array => 4,
                    LogicalType::Float64Array => 8,
                    _ => return Err(anyhow!("rowpack invalid array type")),
                }
            } else {
                match storage {
                    FloatStorage::F16 => 2,
                    FloatStorage::F32 => 4,
                    FloatStorage::F64 => 8,
                }
            };
            let max_len = data.len() / elem_bytes;
            if len > MAX_ARRAY_LEN || len > max_len || len > u32::MAX as usize {
                return Err(anyhow!("rowpack array length overflow"));
            }
            mp::write_array_len(out, len as u32)?;
            for _ in 0..len {
                let v = if tag == 2 {
                    decode_float_fallback(data, col.logical_type)?
                } else {
                    decode_float(data, *storage)?
                };
                match col.logical_type {
                    LogicalType::Float32Array => mp::write_f32(out, v as f32)?,
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
        ColumnEncoding::Int { signed, varint, bits } => {
            if *signed {
                JsonValue::from(decode_int(data, *varint, *bits)?)
            } else {
                JsonValue::from(decode_uint(data, *varint, *bits)?)
            }
        }
        ColumnEncoding::Float { storage } => {
            let v = if tag == 2 {
                decode_float_fallback(data, col.logical_type)?
            } else {
                decode_float(data, *storage)?
            };
            match col.logical_type {
                LogicalType::Float32 => JsonValue::from(v as f32),
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
        ColumnEncoding::Date { varint, bits } => {
            JsonValue::from(decode_int(data, *varint, *bits)?)
        }
        ColumnEncoding::Timestamp => {
            let secs = zigzag_decode(read_varint(data)?);
            let nanos = read_varint(data)? as u32;
            let formatted = format_iso(IsoFormat::Rfc3339, secs, nanos);
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
                    DecimalStorage::Int { varint, bits } => {
                        let v = decode_int(data, *varint, *bits)?;
                        if *scale == 0 {
                            JsonValue::from(v)
                        } else {
                            JsonValue::from(format_decimal_i64(v, *scale))
                        }
                    }
                    DecimalStorage::Float { storage } => {
                        let v = decode_float(data, *storage)?;
                        JsonValue::from(v)
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
            let elem_bytes = if tag == 2 {
                match col.logical_type {
                    LogicalType::Float32Array => 4,
                    LogicalType::Float64Array => 8,
                    _ => return Err(anyhow!("rowpack invalid array type")),
                }
            } else {
                match storage {
                    FloatStorage::F16 => 2,
                    FloatStorage::F32 => 4,
                    FloatStorage::F64 => 8,
                }
            };
            let max_len = data.len() / elem_bytes;
            if len > MAX_ARRAY_LEN || len > max_len {
                return Err(anyhow!("rowpack array length overflow"));
            }
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let v = if tag == 2 {
                    decode_float_fallback(data, col.logical_type)?
                } else {
                    decode_float(data, *storage)?
                };
                match col.logical_type {
                    LogicalType::Float32Array => values.push(JsonValue::from(v as f32)),
                    _ => values.push(JsonValue::from(v)),
                }
            }
            JsonValue::from(values)
        }
        ColumnEncoding::Bool => JsonValue::Null,
    };
    Ok(value)
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

fn decode_float_fallback(data: &mut &[u8], logical: LogicalType) -> Result<f64> {
    match logical {
        LogicalType::Float32 | LogicalType::Float32Array => {
            if data.len() < 4 {
                return Err(anyhow!("rowpack float fallback truncated"));
            }
            let (head, tail) = data.split_at(4);
            *data = tail;
            Ok(f32::from_le_bytes(head.try_into().unwrap()) as f64)
        }
        _ => {
            if data.len() < 8 {
                return Err(anyhow!("rowpack float fallback truncated"));
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

fn format_decimal_i64(value: i64, scale: i32) -> String {
    let negative = value.is_negative();
    let mut digits = value.abs().to_string();
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
            let mut out = Vec::new();
            for _ in 0..token_count {
                let token = read_varint(data)? as usize;
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
        StringEncoding::NumericInt => {
            let value = zigzag_decode(read_varint(data)?);
            Ok(value.to_string())
        }
        StringEncoding::NumericScaled => {
            let value = zigzag_decode(read_varint(data)?);
            let scale = read_varint(data)? as u32;
            if scale > MAX_NUMERIC_SCALE {
                return Err(anyhow!("rowpack numeric scale too large"));
            }
            Ok(format_scaled_i64(value, scale))
        }
        StringEncoding::Hex { len, dashed, upper } => {
            let byte_len = len / 2;
            if data.len() < byte_len {
                return Err(anyhow!("rowpack hex truncated"));
            }
            let (head, tail) = data.split_at(byte_len);
            *data = tail;
            Ok(decode_hex(head, *len, *dashed, *upper))
        }
        StringEncoding::IsoTimestamp { fmt } => {
            let secs = zigzag_decode(read_varint(data)?);
            let nanos = read_varint(data)? as u32;
            Ok(format_iso(*fmt, secs, nanos))
        }
        StringEncoding::Snappy => {
            let len = read_varint(data)? as usize;
            if len > MAX_FIELD_BYTES {
                return Err(anyhow!("rowpack snappy too large"));
            }
            if data.len() < len {
                return Err(anyhow!("rowpack snappy truncated"));
            }
            let (head, tail) = data.split_at(len);
            *data = tail;
            let decoded = snap::raw::Decoder::new().decompress_vec(head)?;
            if decoded.len() > MAX_FIELD_BYTES {
                return Err(anyhow!("rowpack snappy too large"));
            }
            Ok(String::from_utf8_lossy(&decoded).to_string())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn build_context() -> RowPackContext {
        RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![
                ColumnContext {
                    name: "id".to_string(),
                    logical_type: LogicalType::Int64,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: true,
                        bits: 64,
                    },
                },
                ColumnContext {
                    name: "name".to_string(),
                    logical_type: LogicalType::String,
                    encoding: ColumnEncoding::String {
                        encoding: StringEncoding::Dict {
                            values: vec![b"alice".to_vec(), b"bob".to_vec()],
                        },
                    },
                },
                ColumnContext {
                    name: "active".to_string(),
                    logical_type: LogicalType::Bool,
                    encoding: ColumnEncoding::Bool,
                },
                ColumnContext {
                    name: "score".to_string(),
                    logical_type: LogicalType::Float32,
                    encoding: ColumnEncoding::Float {
                        storage: FloatStorage::F16,
                    },
                },
                ColumnContext {
                    name: "payload".to_string(),
                    logical_type: LogicalType::Bytes,
                    encoding: ColumnEncoding::Bytes { fixed_len: None },
                },
                ColumnContext {
                    name: "created_at".to_string(),
                    logical_type: LogicalType::Timestamp,
                    encoding: ColumnEncoding::Timestamp,
                },
                ColumnContext {
                    name: "embed".to_string(),
                    logical_type: LogicalType::Float32Array,
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
        let encoded = encoder.encode_row(&row).unwrap();
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
                logical_type: LogicalType::Timestamp,
                encoding: ColumnEncoding::Timestamp,
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Timestamp {
            secs: 1_700_000_000,
            nanos: 123_000_000,
        }];
        let encoded = encoder.encode_row(&row).unwrap();
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
                logical_type: LogicalType::Decimal128,
                encoding: ColumnEncoding::Decimal {
                    precision: 10,
                    scale: 2,
                    bytes: 16,
                    storage: DecimalStorage::Float {
                        storage: FloatStorage::F64,
                    },
                },
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Decimal128(-12345i128)];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert_eq!(map.get("amount"), Some(&JsonValue::from(-123.45)));
    }

    #[test]
    fn decimal256_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "amount".to_string(),
                logical_type: LogicalType::Decimal256,
                encoding: ColumnEncoding::Decimal {
                    precision: 50,
                    scale: 4,
                    bytes: 32,
                    storage: DecimalStorage::Float {
                        storage: FloatStorage::F64,
                    },
                },
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let big = BigInt::parse_bytes(b"123456789012345678901234567890", 10).unwrap();
        let row = vec![ValueRef::Decimal256(decimal256_bytes(&big))];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        let value = map
            .get("amount")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!(value.is_finite());
    }

    #[test]
    fn decimal_lossless_raw_encoding() {
        let mut config = RowPackConfig::default();
        config.decimal_allow_lossy = Some(false);
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "amount".to_string(),
                logical_type: LogicalType::Decimal128,
                fixed_len: None,
                precision: Some(10),
                scale: Some(2),
            }],
            config,
        );
        builder
            .observe_row(&[ValueRef::Decimal128(-12345i128)])
            .unwrap();
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::Decimal {
                storage: DecimalStorage::Raw,
                ..
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
    }

    #[test]
    fn decimal_int_encoding_scale_zero() {
        let mut config = RowPackConfig::default();
        config.decimal_allow_lossy = Some(true);
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "amount".to_string(),
                logical_type: LogicalType::Decimal128,
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
                storage: DecimalStorage::Int { .. },
                ..
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
    }

    #[test]
    fn decimal_lossy_float_encoding_scale_nonzero() {
        let mut config = RowPackConfig::default();
        config.decimal_allow_lossy = Some(true);
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "amount".to_string(),
                logical_type: LogicalType::Decimal128,
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
                storage: DecimalStorage::Float { .. },
                ..
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Decimal128(123456i128)];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        let value = map.get("amount").and_then(|v| v.as_f64()).unwrap();
        assert!((value - 123.456).abs() < 1e-6);
    }

    #[test]
    fn decimal_lossless_scale_nonzero_raw_roundtrip() {
        let mut config = RowPackConfig::default();
        config.decimal_allow_lossy = Some(false);
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "amount".to_string(),
                logical_type: LogicalType::Decimal128,
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
                storage: DecimalStorage::Raw,
                ..
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Decimal128(123456i128)];
        let encoded = encoder.encode_row(&row).unwrap();
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
                logical_type: LogicalType::Decimal128,
                encoding: ColumnEncoding::Decimal {
                    precision: 20,
                    scale: 0,
                    bytes: 16,
                    storage: DecimalStorage::Int {
                        varint: false,
                        bits: 64,
                    },
                },
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let big = i128::MAX;
        let row = vec![ValueRef::Decimal128(big)];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert!(map.get("amount").unwrap().as_str().is_some());
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
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("carol"));
    }

    #[test]
    fn iso_timestamp_encoding() {
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "ts".to_string(),
                logical_type: LogicalType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            RowPackConfig::default(),
        );
        builder
            .observe_row(&[ValueRef::String("2024-01-01T00:00:00Z")])
            .unwrap();
        let ctx = builder.finish().unwrap();
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("2024-01-01T00:00:00Z")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("2024"));
    }

    #[test]
    fn hex_encoding() {
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "hex".to_string(),
                logical_type: LogicalType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            RowPackConfig::default(),
        );
        builder
            .observe_row(&[ValueRef::String("0123456789abcdef0123456789abcdef")])
            .unwrap();
        let ctx = builder.finish().unwrap();
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("0123456789abcdef0123456789abcdef")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("0123456789abcdef"));
    }

    #[test]
    fn hex_uppercase_encoding() {
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "hex".to_string(),
                logical_type: LogicalType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            RowPackConfig::default(),
        );
        builder
            .observe_row(&[ValueRef::String("ABCDEF0123456789ABCDEF0123456789")])
            .unwrap();
        let ctx = builder.finish().unwrap();
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("ABCDEF0123456789ABCDEF0123456789")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("ABCDEF"));
    }

    #[test]
    fn hex_mixed_case_falls_back() {
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "hex".to_string(),
                logical_type: LogicalType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            RowPackConfig::default(),
        );
        builder
            .observe_row(&[ValueRef::String("Abcdef0123456789ABCDEF0123456789")])
            .unwrap();
        let ctx = builder.finish().unwrap();
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("Abcdef0123456789ABCDEF0123456789")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("Abcdef"));
    }

    #[test]
    fn alphabet_encoding_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "code".to_string(),
                logical_type: LogicalType::String,
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
        let encoded = encoder.encode_row(&row).unwrap();
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
                logical_type: LogicalType::String,
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
        let encoded = encoder.encode_row(&row).unwrap();
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
                logical_type: LogicalType::String,
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
                logical_type: LogicalType::String,
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
                logical_type: LogicalType::String,
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
        let encoded = encoder.encode_row(&row).unwrap();
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
                logical_type: LogicalType::String,
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
                logical_type: LogicalType::String,
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
        write_varint(256 + 10, &mut row);
        let err = decoder.decode_json(&row).unwrap_err();
        assert!(err.to_string().contains("bpe token out of range"));
    }

    #[test]
    fn bpe_decode_rejects_excess_tokens() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "code".to_string(),
                logical_type: LogicalType::String,
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

    #[test]
    fn numeric_int_encoding_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "num".to_string(),
                logical_type: LogicalType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::NumericInt,
                },
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("12345")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert_eq!(map.get("num").and_then(|v| v.as_str()), Some("12345"));
    }

    #[test]
    fn numeric_scaled_encoding_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "num".to_string(),
                logical_type: LogicalType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::NumericScaled,
                },
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("-12.340")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert_eq!(map.get("num").and_then(|v| v.as_str()), Some("-12.340"));
    }

    #[test]
    fn numeric_scaled_encoding_fallback() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "num".to_string(),
                logical_type: LogicalType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::NumericScaled,
                },
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("12.3a")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert_eq!(map.get("num").and_then(|v| v.as_str()), Some("12.3a"));
    }

    #[test]
    fn numeric_encoding_selected() {
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "num".to_string(),
                logical_type: LogicalType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            RowPackConfig::default(),
        );
        for idx in 0..100 {
            let value = format!("{}", idx);
            builder.observe_row(&[ValueRef::String(&value)]).unwrap();
        }
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::String {
                encoding: StringEncoding::NumericInt,
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
    }

    #[test]
    fn numeric_scaled_selected() {
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "num".to_string(),
                logical_type: LogicalType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            RowPackConfig::default(),
        );
        for idx in 0..100 {
            let value = format!("{}.{}", idx, idx % 10);
            builder.observe_row(&[ValueRef::String(&value)]).unwrap();
        }
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::String {
                encoding: StringEncoding::NumericScaled,
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
    }

    #[test]
    fn numeric_rejects_leading_or_trailing_dot() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "num".to_string(),
                logical_type: LogicalType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::NumericScaled,
                },
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String(".5")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx.clone()).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert_eq!(map.get("num").and_then(|v| v.as_str()), Some(".5"));

        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("1.")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        let JsonValue::Object(map) = json else {
            panic!("expected object");
        };
        assert_eq!(map.get("num").and_then(|v| v.as_str()), Some("1."));
    }

    #[test]
    fn hex_dashed_encoding() {
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "hex".to_string(),
                logical_type: LogicalType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            RowPackConfig::default(),
        );
        builder
            .observe_row(&[ValueRef::String(
                "01234567-89ab-cdef-0123-456789abcdef",
            )])
            .unwrap();
        let ctx = builder.finish().unwrap();
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String(
            "01234567-89ab-cdef-0123-456789abcdef",
        )];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json
            .to_string()
            .contains("01234567-89ab-cdef-0123-456789abcdef"));
    }

    #[test]
    fn hex_fallback_non_hex() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "hex".to_string(),
                logical_type: LogicalType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::Hex {
                        len: 32,
                        dashed: false,
                        upper: false,
                    },
                },
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("not-hex")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("not-hex"));
    }

    #[test]
    fn float16_fallback() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "v".to_string(),
                logical_type: LogicalType::Float64,
                encoding: ColumnEncoding::Float {
                    storage: FloatStorage::F16,
                },
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Float64(999.0)];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("999"));
    }

    #[test]
    fn float_array_float16_encoding() {
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "embed".to_string(),
                logical_type: LogicalType::Float32Array,
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
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("0.1"));
    }

    #[test]
    fn float_force_float64_encoding() {
        let mut config = RowPackConfig::default();
        config.force_float64 = Some(true);
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "score".to_string(),
                logical_type: LogicalType::Float32,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            config,
        );
        builder.observe_row(&[ValueRef::Float32(1.5)]).unwrap();
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::Float { storage } => {
                assert_eq!(*storage, FloatStorage::F64);
            }
            other => panic!("unexpected encoding: {:?}", other),
        }
    }

    #[test]
    fn bytes_roundtrip() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "blob".to_string(),
                logical_type: LogicalType::Bytes,
                encoding: ColumnEncoding::Bytes { fixed_len: None },
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Bytes(b"xyz")];
        let encoded = encoder.encode_row(&row).unwrap();
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
                logical_type: LogicalType::Bytes,
                encoding: ColumnEncoding::Bytes { fixed_len: Some(4) },
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Bytes(b"ABCD")];
        let encoded = encoder.encode_row(&row).unwrap();
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
                logical_type: LogicalType::Bytes,
                encoding: ColumnEncoding::Bytes { fixed_len: Some(4) },
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Bytes(b"ABCDE")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("QUJDREU="));
    }

    #[test]
    fn small_int_encodings() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![
                ColumnContext {
                    name: "i8".to_string(),
                    logical_type: LogicalType::Int8,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: false,
                        bits: 8,
                    },
                },
                ColumnContext {
                    name: "u8".to_string(),
                    logical_type: LogicalType::UInt8,
                    encoding: ColumnEncoding::Int {
                        signed: false,
                        varint: false,
                        bits: 8,
                    },
                },
                ColumnContext {
                    name: "i16".to_string(),
                    logical_type: LogicalType::Int16,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: false,
                        bits: 16,
                    },
                },
                ColumnContext {
                    name: "u16".to_string(),
                    logical_type: LogicalType::UInt16,
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
        let encoded = encoder.encode_row(&row).unwrap();
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
                logical_type: LogicalType::Int32,
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
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("-12345"));
    }

    #[test]
    fn iso_fallback_string() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "ts".to_string(),
                logical_type: LogicalType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::IsoTimestamp { fmt: IsoFormat::Rfc3339 },
                },
            }],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("not-a-timestamp")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("not-a-timestamp"));
    }

    #[test]
    fn iso_date_only_encoding() {
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "ts".to_string(),
                logical_type: LogicalType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            RowPackConfig::default(),
        );
        builder.observe_row(&[ValueRef::String("2024-02-03")]).unwrap();
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::String {
                encoding: StringEncoding::IsoTimestamp { fmt },
            } => {
                assert_eq!(*fmt, IsoFormat::DateOnly);
            }
            other => panic!("unexpected encoding: {:?}", other),
        }
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("2024-02-03")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("2024-02-03"));
    }

    #[test]
    fn iso_date_time_space_encoding() {
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "ts".to_string(),
                logical_type: LogicalType::String,
                fixed_len: None,
                precision: None,
                scale: None,
            }],
            RowPackConfig::default(),
        );
        builder
            .observe_row(&[ValueRef::String("2024-02-03 12:34:56")])
            .unwrap();
        let ctx = builder.finish().unwrap();
        match &ctx.columns[0].encoding {
            ColumnEncoding::String {
                encoding: StringEncoding::IsoTimestamp { fmt },
            } => {
                assert_eq!(*fmt, IsoFormat::DateTimeSpace);
            }
            other => panic!("unexpected encoding: {:?}", other),
        }
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String("2024-02-03 12:34:56")];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("2024-02-03 12:34:56"));
    }

    #[test]
    fn snappy_encoding_selected() {
        let mut config = RowPackConfig::default();
        config.string_long_min_len = Some(8);
        config.string_long_min_ratio = Some(0.9);
        config.max_dict_entries = Some(0);
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "text".to_string(),
                logical_type: LogicalType::String,
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
                encoding: StringEncoding::Snappy,
            } => {}
            other => panic!("unexpected encoding: {:?}", other),
        }
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::String(&value)];
        let encoded = encoder.encode_row(&row).unwrap();
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let json = decoder.decode_json(&encoded).unwrap();
        assert!(json.to_string().contains("aaaaaaaa"));
    }

    #[test]
    fn snappy_falls_back_to_raw() {
        let mut config = RowPackConfig::default();
        config.string_long_min_len = Some(8);
        config.string_long_min_ratio = Some(0.5);
        config.max_dict_entries = Some(0);
        let mut builder = RowPackContextBuilder::new(
            vec![RowPackColumn {
                name: "text".to_string(),
                logical_type: LogicalType::String,
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
        let encoded = encoder.encode_row(&row).unwrap();
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
                logical_type: LogicalType::String,
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
        let encoded = encoder.encode_row(&row).unwrap();
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
                    logical_type: LogicalType::Int64,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: true,
                        bits: 64,
                    },
                },
                ColumnContext {
                    name: "note".to_string(),
                    logical_type: LogicalType::String,
                    encoding: ColumnEncoding::String {
                        encoding: StringEncoding::Raw,
                    },
                },
            ],
        estimated_row_bytes: 0,
        };
        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let row = vec![ValueRef::Null, ValueRef::Null];
        let encoded = encoder.encode_row(&row).unwrap();
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
                logical_type: LogicalType::String,
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
                logical_type: LogicalType::Float32Array,
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
                logical_type: LogicalType::Bool,
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
                logical_type: LogicalType::Int64,
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
    fn decode_rejects_corrupt_snappy() {
        let ctx = RowPackContext {
            version: ROWPACK_VERSION,
            columns: vec![ColumnContext {
                name: "text".to_string(),
                logical_type: LogicalType::String,
                encoding: ColumnEncoding::String {
                    encoding: StringEncoding::Snappy,
                },
            }],
        estimated_row_bytes: 0,
        };
        let decoder = RowPackDecoder::new(ctx).unwrap();
        let mut tags = BitWriter::new();
        tags.write_bits(1, 2);
        let mut row = tags.finish();
        row.push(0x05);
        row.extend_from_slice(&[0x00, 0x01, 0x02, 0x03, 0x04]);
        let err = decoder.decode_json(&row).unwrap_err();
        assert!(err.to_string().contains("snappy"));
    }
}
